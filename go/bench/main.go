// bench: load generator replaying laya request fixtures (jsonl: {state, questions}).
//
// Usage: ./bench -u http://127.0.0.1:8303/predict -c 16 -d 20s -f basic.jsonl [-slo 600ms] [-warm 2s]
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type fixture struct {
	State     any                       `json:"state"`
	Questions map[string]map[string]any `json:"questions"`
}

func main() {
	var (
		url  = flag.String("u", "http://127.0.0.1:8303/predict", "target url")
		conc = flag.Int("c", 8, "concurrency")
		dur  = flag.Duration("d", 20*time.Second, "duration")
		file = flag.String("f", "basic.jsonl", "fixture jsonl")
		slo  = flag.Duration("slo", 600*time.Millisecond, "SLO")
		warm = flag.Duration("warm", 2*time.Second, "warmup (not counted)")
	)
	flag.Parse()

	f, err := os.Open(*file)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open fixture:", err)
		os.Exit(1)
	}
	var fixtures []fixture
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	for sc.Scan() {
		var row struct {
			State     any                       `json:"state"`
			Questions map[string]map[string]any `json:"questions"`
		}
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			fmt.Fprintln(os.Stderr, "skip bad line:", err)
			continue
		}
		fixtures = append(fixtures, row)
	}
	f.Close()
	if len(fixtures) == 0 {
		fmt.Fprintln(os.Stderr, "no fixtures")
		os.Exit(1)
	}

	// pre-serialize request bodies
	bodies := make([][]byte, len(fixtures))
	for i, fx := range fixtures {
		bodies[i], _ = json.Marshal(map[string]any{"state": fx.State, "questions": fx.Questions})
	}

	client := &http.Client{
		Timeout: 10 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        *conc * 2,
			MaxIdleConnsPerHost: *conc * 2,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	var (
		idx      atomic.Int64
		ok       atomic.Int64
		failed   atomic.Int64
		sloOK    atomic.Int64
		errOnce  sync.Once
		firstErr string
	)
	latCh := make(chan time.Duration, 1<<20)
	var wg sync.WaitGroup
	deadline := time.Now().Add(*dur)
	warmEnd := time.Now().Add(*warm)

	for w := 0; w < *conc; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for time.Now().Before(deadline) {
				i := int(idx.Add(1)-1) % len(bodies)
				t0 := time.Now()
				resp, err := client.Post(*url, "application/json", bytes.NewReader(bodies[i]))
				lat := time.Since(t0)
				if err != nil {
					failed.Add(1)
					errOnce.Do(func() { firstErr = "client: " + err.Error() })
					continue
				}
				body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
				resp.Body.Close()
				if resp.StatusCode != 200 {
					failed.Add(1)
					errOnce.Do(func() { firstErr = fmt.Sprintf("http %d: %.300s", resp.StatusCode, body) })
					continue
				}
				if time.Now().After(warmEnd) {
					ok.Add(1)
					if lat <= *slo {
						sloOK.Add(1)
					}
					select {
					case latCh <- lat:
					default:
					}
				}
			}
		}()
	}
	wg.Wait()
	close(latCh)

	lats := make([]time.Duration, 0, len(latCh))
	for l := range latCh {
		lats = append(lats, l)
	}
	sort.Slice(lats, func(i, j int) bool { return lats[i] < lats[j] })

	total := ok.Load() + failed.Load()
	rps := float64(ok.Load()) / dur.Seconds()
	pct := func(p float64) time.Duration {
		if len(lats) == 0 {
			return 0
		}
		i := int(float64(len(lats)-1) * p)
		return lats[i]
	}
	sloPct := 100.0
	if ok.Load() > 0 {
		sloPct = 100 * float64(sloOK.Load()) / float64(ok.Load())
	}
	verdict := "FAIL"
	if len(lats) > 0 && sloPct >= 99.0 {
		verdict = "PASS"
	}
	fmt.Printf("c=%d  dur=%s  fixtures=%d\n", *conc, dur, len(bodies))
	fmt.Printf("requests=%d ok=%d failed=%d  rps=%.0f\n", total, ok.Load(), failed.Load(), rps)
	if firstErr != "" {
		fmt.Printf("first_error: %s\n", firstErr)
	}
	if len(lats) > 0 {
		fmt.Printf("latency p50=%s p90=%s p95=%s p99=%s max=%s\n",
			pct(0.50), pct(0.90), pct(0.95), pct(0.99), lats[len(lats)-1])
	}
	fmt.Printf("SLO %s: %.1f%% within -> %s\n", slo, sloPct, verdict)
}
