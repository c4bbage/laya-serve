// layagate: dynamic-batching gateway for laya batch backend.
//
// POST /predict {"id"?, "state", "questions"} -> batching window -> backend /predict_batch -> per-request answers.
//
// Env: BACKEND_URL (default http://127.0.0.1:8302), LISTEN (:8303),
//
//	BATCH_MAX (64), BATCH_WINDOW_MS (10)
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

type pendingReq struct {
	body   map[string]any
	id     string
	respCh chan batchResult
}

type batchResult struct {
	answers map[string]any
	err     error
}

var (
	backendURLs = backendList()
	listenAddr  = envOr("LISTEN", ":8303")
	batchMax    = envInt("BATCH_MAX", 64)
	batchWindow = time.Duration(envInt("BATCH_WINDOW_MS", 10)) * time.Millisecond
	backendSeq  atomic.Int64

	reqCh = make(chan *pendingReq, 8192)
	seq   atomic.Int64

	statTotal    atomic.Int64
	statBatches  atomic.Int64
	statBatchSum atomic.Int64 // sum of batch sizes
	statErrors   atomic.Int64
)

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func envInt(k string, d int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return d
}

func backendList() []string {
	raw := envOr("BACKENDS", envOr("BACKEND_URL", "http://127.0.0.1:8302"))
	var out []string
	for _, s := range strings.Split(raw, ",") {
		if s = strings.TrimSpace(s); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func nextBackend() string {
	return backendURLs[int(backendSeq.Add(1)-1)%len(backendURLs)]
}

var httpClient = &http.Client{
	Timeout: 5 * time.Second,
	Transport: &http.Transport{
		MaxIdleConns:        256,
		MaxIdleConnsPerHost: 256,
		IdleConnTimeout:     90 * time.Second,
	},
}

func batcher() {
	for {
		first := <-reqCh
		batch := []*pendingReq{first}
		deadline := time.Now().Add(batchWindow)
		for len(batch) < batchMax {
			remaining := time.Until(deadline)
			if remaining <= 0 {
				// drain whatever is immediately available
				for len(batch) < batchMax {
					select {
					case r := <-reqCh:
						batch = append(batch, r)
						continue
					default:
					}
					break
				}
				break
			}
			select {
			case r := <-reqCh:
				batch = append(batch, r)
			case <-time.After(remaining):
			}
		}

		statBatches.Add(1)
		statBatchSum.Add(int64(len(batch)))

		go dispatch(batch)
	}
}

func dispatch(batch []*pendingReq) {
	reqs := make([]map[string]any, 0, len(batch))
	for _, r := range batch {
		reqs = append(reqs, r.body)
	}
	payload, err := json.Marshal(map[string]any{"requests": reqs})

	var answers map[string]any
	backend := nextBackend()
	if err == nil {
		var resp *http.Response
		resp, err = httpClient.Post(backend+"/predict_batch", "application/json", bytes.NewReader(payload))
		if err == nil {
			var out struct {
				Answers map[string]any `json:"answers"`
			}
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode == 200 {
				if jerr := json.Unmarshal(body, &out); jerr == nil {
					answers = out.Answers
				} else {
					err = fmt.Errorf("bad backend body: %v", jerr)
				}
			} else {
				err = fmt.Errorf("backend %d: %.200s", resp.StatusCode, body)
			}
		}
	}

	if err != nil {
		statErrors.Add(int64(len(batch)))
	}
	for _, r := range batch {
		if err != nil {
			r.respCh <- batchResult{err: err}
		} else if a, ok := answers[r.id]; ok {
			r.respCh <- batchResult{answers: a.(map[string]any)}
		} else {
			r.respCh <- batchResult{err: fmt.Errorf("missing answers for %s", r.id)}
		}
	}
}

func handlePredict(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "bad json: "+err.Error(), 400)
		return
	}
	id, _ := body["id"].(string)
	if id == "" {
		id = "r" + strconv.FormatInt(seq.Add(1), 10)
		body["id"] = id
	}
	pr := &pendingReq{body: body, id: id, respCh: make(chan batchResult, 1)}
	statTotal.Add(1)

	select {
	case reqCh <- pr:
	default:
		http.Error(w, "gateway queue full", 503)
		return
	}

	res := <-pr.respCh
	if res.err != nil {
		http.Error(w, res.err.Error(), 502)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(res.answers)
}

func handleStats(w http.ResponseWriter, _ *http.Request) {
	total := statTotal.Load()
	batches := statBatches.Load()
	avg := 0.0
	if batches > 0 {
		avg = float64(statBatchSum.Load()) / float64(batches)
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"requests_total": total,
		"batches":        batches,
		"avg_batch_size": avg,
		"errors":         statErrors.Load(),
	})
}

func main() {
	go batcher()
	mux := http.NewServeMux()
	mux.HandleFunc("/predict", handlePredict)
	mux.HandleFunc("/stats", handleStats)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok")) })
	log.Printf("layagate listening %s -> %v (batch_max=%d window=%s)", listenAddr, backendURLs, batchMax, batchWindow)
	log.Fatal(http.ListenAndServe(listenAddr, mux))
}
