// paritycheck validates the pure-Go laya stack against Python dumps.
//
//	-mode tokens  : tokenizer + render_options vs laya_token_probe.json
//	-mode seq     : full BuildSequence vs laya_parity.jsonl (ids + markers, must be 100%)
//	-mode full    : + ONNX inference vs dumped bf16 answers (argmax agreement + prob delta)
//
// Env for -mode full: LD_LIBRARY_PATH must include onnxruntime + nvidia lib dirs.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"

	"layatools/layago"
)

func die(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

type flags map[string]string

func parseFlags() flags {
	f := flags{}
	for i := 1; i < len(os.Args)-1; i++ {
		a := os.Args[i]
		if len(a) > 2 && a[0] == '-' && a[1] != '-' {
			f[a[1:]] = os.Args[i+1]
			i++
		}
	}
	return f
}

func (f flags) get(k, def string) string {
	if v, ok := f[k]; ok {
		return v
	}
	return def
}

func decodeOrderedBytes(b []byte) (any, error) {
	return layago.DecodeOrdered(bytes.NewReader(b))
}

func main() {
	fl := parseFlags()
	mode := fl.get("mode", "seq")
	dir := fl.get("dir", ".")

	tok, err := layago.LoadTokenizer(dir+"/laya_vocab.json", dir+"/laya_merges.json")
	if err != nil {
		die("load tokenizer: %v", err)
	}
	cfg, err := layago.LoadCfg(dir + "/laya_go_cfg.json")
	if err != nil {
		die("load cfg: %v", err)
	}

	switch mode {
	case "tokens":
		checkTokens(dir, tok, cfg)
	case "seq":
		checkSeq(dir, fl.get("fixtures", dir+"/basic.jsonl"), tok, cfg, nil)
	case "full":
		if err := layago.Init(fl.get("ortlib", os.Getenv("ORT_LIB"))); err != nil {
			die("ort init: %v", err)
		}
		var trt *layago.TRTOpts
		if fl.get("provider", os.Getenv("GOSERVE_PROVIDER")) == "trt" {
			trt = &layago.TRTOpts{
				DeviceID:  fl.get("gpu", ""),
				CachePath: os.Getenv("TRT_CACHE"),
				Precision: os.Getenv("TRT_PRECISION"),
				LNFP32:    os.Getenv("TRT_LN_FP32") == "1",
				Workspace: envIntOr("TRT_WORKSPACE", 0),
			}
		}
		eng, err := layago.NewEngineTRT(fl.get("onnx", dir+"/laya.onnx"), fl.get("gpu", ""), envIntOr("GOSERVE_MEM_LIMIT", 0), trt)
		if err != nil {
			die("engine: %v", err)
		}
		checkSeq(dir, fl.get("fixtures", dir+"/basic.jsonl"), tok, cfg, eng)
	default:
		die("unknown mode %s", mode)
	}
}

// ---------- tokens ----------

func checkTokens(dir string, tok *layago.Tokenizer, cfg *layago.Cfg) {
	raw, err := os.ReadFile(dir + "/laya_token_probe.json")
	if err != nil {
		die("open probe: %v", err)
	}
	v, err := decodeOrderedBytes(raw)
	if err != nil {
		die("decode probe: %v", err)
	}
	entries := v.([]any)
	nHead, nOpt, nState, bad := 0, 0, 0, 0
	for _, eAny := range entries {
		e := eAny.(*layago.Obj)
		state := e.M["state"]
		wantState := idsOf(e.M["state_ids"])
		gotState := tok.Encode(layago.ReplaceMask(layago.SerializeState(state), cfg.MaskToken))
		nState++
		if !eq(gotState, wantState) {
			bad++
			fmt.Printf("STATE MISMATCH %s: got %d want %d ids\n", str(e.M["id"]), len(gotState), len(wantState))
		}
		qs := e.M["questions"].(*layago.Obj)
		for _, qid := range qs.Keys {
			qd := qs.M[qid].(*layago.Obj)
			q, err := layago.ToInternal(qd)
			if err != nil {
				die("toInternal: %v", err)
			}
			wantHead := idsOf(qd.M["head"])
			gotHead := tok.Encode(layago.ReplaceMask(q.T+" question: "+q.Ins, cfg.MaskToken))
			nHead++
			if !eq(gotHead, wantHead) {
				bad++
				fmt.Printf("HEAD MISMATCH %s/%s: got %v want %v\n", str(e.M["id"]), qid, gotHead[:min(8, len(gotHead))], wantHead[:min(8, len(wantHead))])
			}
			wantOpts := qd.M["opts"].([]any)
			opts := layago.RenderOptions(q)
			for i, want := range wantOpts {
				got := tok.Encode(layago.ReplaceMask(" "+opts[i], cfg.MaskToken))
				if len(got) > 48 {
					got = got[:48]
				}
				nOpt++
				if !eq(got, idsOf(want)) {
					bad++
					fmt.Printf("OPT MISMATCH %s/%s opt%d: got %v want %v\n", str(e.M["id"]), qid, i, got, idsOf(want))
				}
			}
		}
	}
	fmt.Printf("tokens: %d states, %d heads, %d opts checked, %d mismatch\n", nState, nHead, nOpt, bad)
	if bad > 0 {
		os.Exit(1)
	}
}

// ---------- seq / full ----------

type parityQ struct {
	IDs     []int32 `json:"ids"`
	Markers []int32 `json:"markers"`
}
type parityEntry struct {
	ID        string                     `json:"id"`
	Questions map[string]parityQ         `json:"questions"`
	Answers   map[string]json.RawMessage `json:"answers"`
}

type ref struct {
	q    *layago.QInternal
	want map[string]any
}

func checkSeq(dir, fixtures string, tok *layago.Tokenizer, cfg *layago.Cfg, eng *layago.Engine) {
	pf, err := os.Open(dir + "/laya_parity.jsonl")
	if err != nil {
		die("open parity: %v", err)
	}
	defer pf.Close()
	parity := map[string]*parityEntry{}
	sc := bufio.NewScanner(pf)
	sc.Buffer(make([]byte, 8*1024*1024), 8*1024*1024)
	for sc.Scan() {
		var e parityEntry
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			die("parity line: %v", err)
		}
		parity[e.ID] = &e
	}
	ff, err := os.Open(fixtures)
	if err != nil {
		die("open fixtures: %v", err)
	}
	defer ff.Close()

	nQ, nBad := 0, 0
	var items []*layago.Item
	var refs []ref
	fs := bufio.NewScanner(ff)
	fs.Buffer(make([]byte, 8*1024*1024), 8*1024*1024)
	for fs.Scan() {
		var fx struct {
			ID        string          `json:"id"`
			State     json.RawMessage `json:"state"`
			Questions json.RawMessage `json:"questions"`
		}
		if err := json.Unmarshal(fs.Bytes(), &fx); err != nil {
			die("fixture: %v", err)
		}
		p := parity[fx.ID]
		if p == nil {
			continue
		}
		stateV, err := decodeOrderedBytes(fx.State)
		if err != nil {
			die("state decode: %v", err)
		}
		qsV, err := decodeOrderedBytes(fx.Questions)
		if err != nil {
			die("questions decode: %v", err)
		}
		qobj := qsV.(*layago.Obj)
		for _, qid := range qobj.Keys {
			q, err := layago.ToInternal(qobj.M[qid].(*layago.Obj))
			if err != nil {
				die("toInternal %s/%s: %v", fx.ID, qid, err)
			}
			it, err := layago.BuildSequence(tok, cfg, stateV, q)
			if err != nil {
				die("build %s/%s: %v", fx.ID, qid, err)
			}
			want := p.Questions[qid]
			if want.IDs == nil {
				die("parity missing %s/%s", fx.ID, qid)
			}
			nQ++
			if !eq(it.IDs, want.IDs) || !eq(it.Markers, want.Markers) {
				nBad++
				if nBad <= 3 {
					fmt.Printf("SEQ MISMATCH %s/%s: ids %d vs %d, markers %d vs %d\n",
						fx.ID, qid, len(it.IDs), len(want.IDs), len(it.Markers), len(want.Markers))
					if os.Getenv("PARITY_VERBOSE") != "" {
						fmt.Printf("  go   %v %v\n  want %v %v\n", it.IDs, it.Markers, want.IDs, want.Markers)
					}
				}
			}
			items = append(items, it)
			var w map[string]any
			json.Unmarshal(p.Answers[qid], &w)
			refs = append(refs, ref{q: q, want: w})
		}
	}
	fmt.Printf("seq: %d questions, %d mismatch\n", nQ, nBad)
	if nBad > 0 {
		os.Exit(1)
	}
	if eng == nil {
		return
	}

	const batch = 32
	agree, total := 0, 0
	var maxDelta, sumDelta float64
	for start := 0; start < len(items); start += batch {
		end := start + batch
		if end > len(items) {
			end = len(items)
		}
		logits, act, _, K, err := eng.Run(items[start:end], cfg.PadTokenID)
		if err != nil {
			die("run: %v", err)
		}
		for r := start; r < end; r++ {
			rr := refs[r]
			ans := layago.AnswerOne(cfg, rr.q, logits[(r-start)*K:(r-start)*K+K], act[(r-start)*2:(r-start)*2+2])
			wantChoice, _ := rr.want["choice"].(string)
			if rr.q.T == "choice" && wantChoice != "" {
				total++
				if ans.Choice == wantChoice {
					agree++
				}
				if wp, ok := rr.want["probabilities"].(map[string]any); ok {
					for k, v := range wp {
						if got, ok2 := ans.Probabilities.M[k]; ok2 {
							wantF, _ := v.(float64)
							d := math.Abs(got.(float64) - wantF)
							if d > maxDelta {
								maxDelta = d
							}
							sumDelta += d
						}
					}
				}
			}
		}
	}
	fmt.Printf("full: choice argmax agreement %d/%d = %.2f%%, max|Δprob|=%.4f, mean|Δprob|=%.4f (vs Python bf16)\n",
		agree, total, 100*float64(agree)/math.Max(1, float64(total)), maxDelta, sumDelta/math.Max(1, float64(total)))
}

// ---------- helpers ----------

func idsOf(v any) []int32 {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]int32, len(arr))
	for i, x := range arr {
		switch n := x.(type) {
		case json.Number:
			f, _ := n.Float64()
			out[i] = int32(f)
		case float64:
			out[i] = int32(n)
		}
	}
	return out
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func eq(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func envIntOr(k string, def int64) int64 {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n
		}
	}
	return def
}
