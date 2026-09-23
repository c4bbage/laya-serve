// goserve: single-binary laya inference server (tokenizer + ONNX CUDA + dynamic batching).
//
// POST /predict {"id"?, "state", "questions": {qid: {"type","instructions","criteria"}}}
//
//	-> {qid: {type, choice, probabilities, confidence, act_probability}}
//
// Env: ONNX_PATH, GOSERVE_GPUS ("0,1"), LISTEN (:8320), BATCH_MAX (48),
//
//	BATCH_WINDOW_MS (10), BUILD_WORKERS (NumCPU)
package main

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"layatools/layago"
)

type pendingReq struct {
	body   *layago.Obj // {id?, state, questions}
	id     string
	qs     *layago.Obj
	state  any
	respCh chan result
}

type result struct {
	answers *layago.Obj
	err     error
}

var (
	onnxPath    = envOr("ONNX_PATH", "laya.onnx")
	gpus        = envOr("GOSERVE_GPUS", "0,1")
	listenAddr  = envOr("LISTEN", ":8320")
	batchMax    = envInt("BATCH_MAX", 48)
	batchWindow = time.Duration(envInt("BATCH_WINDOW_MS", 10)) * time.Millisecond
	buildWorker = envInt("BUILD_WORKERS", runtime.NumCPU())
	memLimit    = envIntOr("GOSERVE_MEM_LIMIT", 6*1024*1024*1024)
	memLimits   = splitCommaInt(os.Getenv("GOSERVE_MEM_LIMIT"))
	provider    = envOr("GOSERVE_PROVIDER", "cuda")

	cfg         *layago.Cfg
	tok         *layago.Tokenizer
	engines     []*layago.Engine
	engineGPUs  []string
	gpuGroups   [][]int // engine indices grouped by physical GPU
	engInflight []atomic.Int64
	engSeq      atomic.Int64
	gpuSeq      atomic.Int64

	reqCh = make(chan *pendingReq, 8192)
	seq   atomic.Int64

	statReq, statBatch, statBatchSum, statErr atomic.Int64
)

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

// batcher groups requests arriving within batchWindow into one engine call.
// batchMax caps the total number of questions (the TRT profile's batch axis
// counts questions, not requests); a request that would overflow the cap is
// carried over to the next batch instead of being appended.
func batcher() {
	var carry *pendingReq
	for {
		first := carry
		carry = nil
		if first == nil {
			first = <-reqCh
		}
		batch := []*pendingReq{first}
		totalQ := len(first.qs.Keys)
		deadline := time.Now().Add(batchWindow)
	fill:
		for totalQ < batchMax {
			var r *pendingReq
			if remaining := time.Until(deadline); remaining > 0 {
				select {
				case r = <-reqCh:
				case <-time.After(remaining):
					break fill
				}
			} else {
				select {
				case r = <-reqCh:
				default:
					break fill
				}
			}
			if totalQ+len(r.qs.Keys) > batchMax {
				carry = r
				break
			}
			batch = append(batch, r)
			totalQ += len(r.qs.Keys)
		}
		statBatch.Add(1)
		statBatchSum.Add(int64(len(batch)))
		go dispatch(batch)
	}
}

type builtQ struct {
	req *pendingReq
	qid string
	q   *layago.QInternal
	it  *layago.Item
	vi  int // index into the valid batch sent to the engine
}

func dispatch(batch []*pendingReq) {
	// build all sequences (parallel across CPUs)
	var built []builtQ
	var mu sync.Mutex
	var wg sync.WaitGroup
	ch := make(chan *pendingReq, len(batch))
	for _, r := range batch {
		ch <- r
	}
	close(ch)
	n := buildWorker
	if n > len(batch) {
		n = len(batch)
	}
	for w := 0; w < n; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var local []builtQ
			for r := range ch {
				for _, qid := range r.qs.Keys {
					q, err := layago.ToInternal(r.qs.M[qid].(*layago.Obj))
					if err != nil {
						mu.Lock()
						local = append(local, builtQ{req: r, qid: qid})
						mu.Unlock()
						continue
					}
					it, err := layago.BuildSequence(tok, cfg, r.state, q)
					if err == nil {
						local = append(local, builtQ{req: r, qid: qid, q: q, it: it})
					} else {
						local = append(local, builtQ{req: r, qid: qid})
					}
				}
			}
			mu.Lock()
			built = append(built, local...)
			mu.Unlock()
		}()
	}
	wg.Wait()

	// any request with zero valid questions errors out
	valid := make([]*layago.Item, 0, len(built))
	// keyed by request, not by client-supplied id: two in-flight requests may share an id
	answers := map[*pendingReq]*layago.Obj{}
	for i := range built {
		b := &built[i]
		if answers[b.req] == nil {
			answers[b.req] = &layago.Obj{M: map[string]any{}}
		}
		if b.it != nil {
			b.vi = len(valid)
			valid = append(valid, b.it)
		}
	}

	if len(valid) > 0 {
		grp := gpuGroups[int(gpuSeq.Add(1)-1)%len(gpuGroups)]
		ei := grp[0]
		if len(grp) > 1 {
			for _, cand := range grp[1:] {
				if engInflight[cand].Load() < engInflight[ei].Load() {
					ei = cand
				}
			}
		}
		eng := engines[ei]
		engInflight[ei].Add(1)
		logits, act, _, K, err := eng.Run(valid, cfg.PadTokenID)
		engInflight[ei].Add(-1)
		if err != nil {
			statErr.Add(int64(len(batch)))
			for _, r := range batch {
				r.respCh <- result{err: err}
			}
			return
		}
		for i := range built {
			b := &built[i]
			if b.it == nil {
				continue
			}
			row := logits[b.vi*K : b.vi*K+K]
			actRow := act[b.vi*2 : b.vi*2+2]
			answers[b.req].Set(b.qid, layago.AnswerOne(cfg, b.q, row, actRow))
		}
	}
	var failed int
	for _, r := range batch {
		if a := answers[r]; a != nil && len(a.Keys) > 0 {
			r.respCh <- result{answers: a}
		} else {
			failed++
			r.respCh <- result{err: errNoAnswers}
		}
	}
	if failed > 0 {
		statErr.Add(int64(failed))
	}
}

var errNoAnswers = &httpError{"no valid questions", 400}

type httpError struct {
	msg  string
	code int
}

func (e *httpError) Error() string { return e.msg }

func handlePredict(w http.ResponseWriter, r *http.Request) {
	v, err := layago.DecodeOrdered(r.Body)
	if err != nil {
		http.Error(w, "bad json: "+err.Error(), 400)
		return
	}
	body, ok := v.(*layago.Obj)
	if !ok {
		http.Error(w, "body must be object", 400)
		return
	}
	id, _ := body.M["id"].(string)
	if id == "" {
		id = "r" + strconv.FormatInt(seq.Add(1), 10)
	}
	qs, ok := body.M["questions"].(*layago.Obj)
	if !ok || len(qs.Keys) == 0 {
		http.Error(w, "questions required", 400)
		return
	}
	if len(qs.Keys) > batchMax {
		http.Error(w, "too many questions in one request (max BATCH_MAX="+strconv.Itoa(batchMax)+")", 400)
		return
	}
	state := body.M["state"]
	if state == nil {
		http.Error(w, "state required", 400)
		return
	}
	pr := &pendingReq{body: body, id: id, qs: qs, state: state, respCh: make(chan result, 1)}
	statReq.Add(1)
	select {
	case reqCh <- pr:
	default:
		http.Error(w, "gateway queue full", 503)
		return
	}
	res := <-pr.respCh
	if res.err != nil {
		code := 502
		if he, ok := res.err.(*httpError); ok {
			code = he.code
		}
		http.Error(w, res.err.Error(), code)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	b, _ := json.Marshal(res.answers)
	w.Write(b)
}

func handleStats(w http.ResponseWriter, _ *http.Request) {
	batches := statBatch.Load()
	avg := 0.0
	if batches > 0 {
		avg = float64(statBatchSum.Load()) / float64(batches)
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"requests_total": statReq.Load(),
		"batches":        batches,
		"avg_batch_size": avg,
		"errors":         statErr.Load(),
	})
}

func main() {
	var err error
	if err = layago.Init(envOr("ORT_LIB", "")); err != nil {
		log.Fatal(err)
	}
	tok, err = layago.LoadTokenizer("laya_vocab.json", "laya_merges.json")
	if err != nil {
		log.Fatal(err)
	}
	cfg, err = layago.LoadCfg("laya_go_cfg.json")
	if err != nil {
		log.Fatal(err)
	}
	for i, g := range splitComma(gpus) {
		ml := memLimit
		if i < len(memLimits) {
			ml = memLimits[i]
		}
		var trt *layago.TRTOpts
		if provider == "trt" {
			trt = &layago.TRTOpts{
				DeviceID:  g,
				SeqMin:    envInt("TRT_SEQ_MIN", 16),
				SeqOpt:    envInt("TRT_SEQ_OPT", 512),
				SeqMax:    envInt("TRT_SEQ_MAX", 1024),
				KMax:      envInt("TRT_KMAX", 32),
				CachePath: envOr("TRT_CACHE", "trtcache"),
				Precision: envOr("TRT_PRECISION", "fp16"),
				LNFP32:    os.Getenv("TRT_LN_FP32") == "1",
				Workspace: envIntOr("TRT_WORKSPACE", 0),
			}
		}
		e, err := layago.NewEngineTRT(onnxPath, g, ml, trt)
		if err != nil {
			log.Fatalf("engine gpu %s: %v", g, err)
		}
		engines = append(engines, e)
		engineGPUs = append(engineGPUs, g)
	}
	byGPU := map[string][]int{}
	for i, g := range engineGPUs {
		byGPU[g] = append(byGPU[g], i)
	}
	for _, g := range engineGPUs {
		if idxs, ok := byGPU[g]; ok {
			gpuGroups = append(gpuGroups, idxs)
			delete(byGPU, g)
		}
	}
	engInflight = make([]atomic.Int64, len(engines))
	go batcher()
	mux := http.NewServeMux()
	mux.HandleFunc("/predict", handlePredict)
	mux.HandleFunc("/stats", handleStats)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, "ok") })
	log.Printf("goserve listening %s onnx=%s gpus=%s batch_max=%d window=%s",
		listenAddr, onnxPath, gpus, batchMax, batchWindow)
	log.Fatal(http.ListenAndServe(listenAddr, mux))
}

func splitCommaInt(s string) []int64 {
	var out []int64
	cur := ""
	for _, r := range s {
		if r == ',' {
			if cur != "" {
				if n, err := strconv.ParseInt(cur, 10, 64); err == nil {
					out = append(out, n)
				}
			}
			cur = ""
		} else {
			cur += string(r)
		}
	}
	if cur != "" {
		if n, err := strconv.ParseInt(cur, 10, 64); err == nil {
			out = append(out, n)
		}
	}
	return out
}

func splitComma(s string) []string {
	var out []string
	cur := ""
	for _, r := range s {
		if r == ',' {
			if cur != "" {
				out = append(out, cur)
			}
			cur = ""
		} else {
			cur += string(r)
		}
	}
	if cur != "" {
		out = append(out, cur)
	}
	if len(out) == 0 {
		out = []string{""}
	}
	return out
}

func envIntOr(k string, def int64) int64 {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n
		}
	}
	return def
}
