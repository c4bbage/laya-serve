package layago

import (
	"fmt"
	"math"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"
)

// Init initializes the ONNX Runtime environment. Must be called once before
// NewEngine. libPath: path to libonnxruntime.so (or "" to use LD_LIBRARY_PATH).
func Init(libPath string) error {
	if libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	return ort.InitializeEnvironment()
}

// Engine wraps one ONNX Runtime CUDA session (one GPU).
type Engine struct {
	sess *ort.DynamicAdvancedSession
	mu   chan struct{} // serializes Run on this GPU
}

// TRTOpts enables the TensorRT execution provider (with CUDA fallback).
// Dynamic-shape profiles cover batch/seq/kmax axes.
type TRTOpts struct {
	DeviceID  string
	SeqMin    int
	SeqOpt    int
	SeqMax    int
	KMax      int
	CachePath string
	Precision string // "fp16" (default) | "bf16" | "fp32"
	LNFP32    bool   // keep LayerNorm in fp32 (trt_layer_norm_fp32_fallback)
	Workspace int64  // builder workspace cap in bytes (0 = TRT default)
}

func (t *TRTOpts) shapes(b, s, k int) string {
	return fmt.Sprintf("input_ids:%dx%d,attention_mask:%dx%d,marker_pos:%dx%d,marker_mask:%dx%d,qtype:%d",
		b, s, b, s, b, k, b, k, b)
}

func defaults(v, def int) int {
	if v <= 0 {
		return def
	}
	return v
}

func appendTRT(so *ort.SessionOptions, t *TRTOpts) error {
	tro, err := ort.NewTensorRTProviderOptions()
	if err != nil {
		return err
	}
	opts := map[string]string{
		"trt_fp16_enable":         "0",
		"trt_bf16_enable":         "0",
		"trt_profile_min_shapes":  t.shapes(1, defaults(t.SeqMin, 16), 1),
		"trt_profile_opt_shapes":  t.shapes(16, defaults(t.SeqOpt, 512), 8),
		"trt_profile_max_shapes":  t.shapes(64, defaults(t.SeqMax, 1024), defaults(t.KMax, 32)),
		"trt_engine_cache_enable": "1",
		"trt_timing_cache_enable": "1",
	}
	switch t.Precision {
	case "", "fp16":
		opts["trt_fp16_enable"] = "1"
	case "bf16":
		opts["trt_bf16_enable"] = "1"
	case "fp32":
	default:
		tro.Destroy()
		return fmt.Errorf("unknown TRT precision %q", t.Precision)
	}
	if t.LNFP32 {
		opts["trt_layer_norm_fp32_fallback"] = "1"
	}
	if t.Workspace > 0 {
		opts["trt_max_workspace_size"] = strconv.FormatInt(t.Workspace, 10)
	}
	if t.DeviceID != "" {
		opts["device_id"] = t.DeviceID
	}
	if t.CachePath != "" {
		opts["trt_engine_cache_path"] = t.CachePath
		opts["trt_timing_cache_path"] = t.CachePath
	}
	if err := tro.Update(opts); err != nil {
		tro.Destroy()
		return err
	}
	err = so.AppendExecutionProviderTensorRT(tro)
	tro.Destroy()
	return err
}

// NewEngine. libPath: path to libonnxruntime.so (or "" to use LD_LIBRARY_PATH).
// Pass trt != nil to use the TensorRT EP (CUDA EP is always appended as fallback).
func NewEngineTRT(onnxPath, gpuID string, memLimitBytes int64, trt *TRTOpts) (*Engine, error) {
	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	if trt != nil {
		if err := appendTRT(so, trt); err != nil {
			so.Destroy()
			return nil, err
		}
	}
	co, err := ort.NewCUDAProviderOptions()
	if err != nil {
		so.Destroy()
		return nil, err
	}
	opts := map[string]string{"arena_extend_strategy": "kSameAsRequested"}
	if gpuID != "" {
		opts["device_id"] = gpuID
	}
	if memLimitBytes > 0 {
		opts["gpu_mem_limit"] = strconv.FormatInt(memLimitBytes, 10)
	}
	if err := co.Update(opts); err != nil {
		co.Destroy()
		so.Destroy()
		return nil, err
	}
	if err := so.AppendExecutionProviderCUDA(co); err != nil {
		co.Destroy()
		so.Destroy()
		return nil, err
	}
	co.Destroy()
	sess, err := ort.NewDynamicAdvancedSession(onnxPath,
		[]string{"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"},
		[]string{"logits", "act_logits"}, so)
	so.Destroy()
	if err != nil {
		return nil, err
	}
	return &Engine{sess: sess, mu: make(chan struct{}, 1)}, nil
}

// NewEngine (CUDA EP only, backward compatible).
func NewEngine(onnxPath, gpuID string, memLimitBytes int64) (*Engine, error) {
	return NewEngineTRT(onnxPath, gpuID, memLimitBytes, nil)
}

func (e *Engine) Destroy() { e.sess.Destroy() }

// Run executes one batched forward pass. Returns logits (n*K) and act (n*2),
// plus n, K for indexing.
func (e *Engine) Run(items []*Item, padID int32) (logits []float32, act []float32, n, K int, err error) {
	e.mu <- struct{}{}
	defer func() { <-e.mu }()

	n = len(items)
	if n == 0 {
		return nil, nil, 0, 0, fmt.Errorf("empty batch")
	}
	L := 0
	K = 0
	for _, it := range items {
		if len(it.IDs) > L {
			L = len(it.IDs)
		}
		if len(it.Markers) > K {
			K = len(it.Markers)
		}
	}
	if K == 0 {
		K = 1
	}

	ids := make([]int64, n*L)
	att := make([]int64, n*L)
	mpos := make([]int64, n*K)
	mmask := make([]int64, n*K)
	qtype := make([]int64, n)
	for i, it := range items {
		copy(ids[i*L:i*L+len(it.IDs)], toInt64(it.IDs))
		for j := 0; j < len(it.IDs); j++ {
			att[i*L+j] = 1
		}
		for j, m := range it.Markers {
			mpos[i*K+j] = int64(m)
			mmask[i*K+j] = 1
		}
		qtype[i] = int64(it.QType)
	}

	tIDs, err := ort.NewTensor(ort.NewShape(int64(n), int64(L)), ids)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	defer tIDs.Destroy()
	tAtt, err := ort.NewTensor(ort.NewShape(int64(n), int64(L)), att)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	defer tAtt.Destroy()
	tPos, err := ort.NewTensor(ort.NewShape(int64(n), int64(K)), mpos)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	defer tPos.Destroy()
	tMask, err := ort.NewTensor(ort.NewShape(int64(n), int64(K)), mmask)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	defer tMask.Destroy()
	tType, err := ort.NewTensor(ort.NewShape(int64(n)), qtype)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	defer tType.Destroy()

	outLogits := make([]float32, n*K)
	tOut, err := ort.NewTensor(ort.NewShape(int64(n), int64(K)), outLogits)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	defer tOut.Destroy()
	outAct := make([]float32, n*2)
	tAct, err := ort.NewTensor(ort.NewShape(int64(n), int64(2)), outAct)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	defer tAct.Destroy()

	if err := e.sess.Run([]ort.Value{tIDs, tAtt, tPos, tMask, tType}, []ort.Value{tOut, tAct}); err != nil {
		return nil, nil, 0, 0, err
	}
	return outLogits, outAct, n, K, nil
}

func toInt64(xs []int32) []int64 {
	out := make([]int64, len(xs))
	for i, x := range xs {
		out[i] = int64(x)
	}
	return out
}

// Answer is the per-question response (mirrors Python stack shape).
type Answer struct {
	Type           string   `json:"type"`
	Choice         string   `json:"choice,omitempty"`
	Probabilities  *Obj     `json:"probabilities,omitempty"`
	Confidence     float64  `json:"confidence"`
	ActProbability float64  `json:"act_probability"`
	Score          *float64 `json:"score,omitempty"`
	Noul           *float64 `json:"noul,omitempty"`
}

func round4(x float64) float64 {
	return math.Round(x*10000) / 10000
}

// AnswerOne postprocesses one row (mirrors agent postprocess with t_scale).
func AnswerOne(cfg *Cfg, q *QInternal, row []float32, actRow []float32) *Answer {
	opts := RenderOptions(q)
	k := len(opts)
	if k == 0 {
		k = 1
	}
	tScale := cfg.TempScale(cfg.QTypes[q.T], k)
	z := make([]float64, k)
	for i := 0; i < k; i++ {
		z[i] = float64(row[i]) / tScale
	}
	p := softmax(z)

	actProb := 0.0
	if len(actRow) >= 2 {
		actProb = round4(softmaxF64(float64(actRow[0]), float64(actRow[1]))[0])
	}
	a := &Answer{Type: q.T, Confidence: round4(confidence(p)), ActProbability: actProb}
	switch q.T {
	case "choice":
		best := 0
		for i := 1; i < k; i++ {
			if p[i] > p[best] {
				best = i
			}
		}
		keys := q.critKeys()
		a.Choice = keys[best]
		po := &Obj{M: map[string]any{}}
		for i, key := range keys {
			po.Keys = append(po.Keys, key)
			po.M[key] = round4(p[i])
		}
		a.Probabilities = po
	case "score":
		var exp float64
		po := &Obj{M: map[string]any{}}
		for i := 0; i < k; i++ {
			exp += float64(i) * p[i]
			key := strconv.Itoa(i)
			po.Keys = append(po.Keys, key)
			po.M[key] = round4(p[i])
		}
		exp = round4(exp)
		a.Score = &exp
		a.Probabilities = po
	case "noul":
		yes := round4(p[1])
		a.Noul = &yes
		a.Confidence = round4(math.Max(p[1], 1-p[1]))
	}
	return a
}

func (q *QInternal) critKeys() []string {
	if q.Crit == nil {
		return nil
	}
	return q.Crit.Keys
}

func softmax(z []float64) []float64 {
	mx := z[0]
	for _, v := range z {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	e := make([]float64, len(z))
	for i, v := range z {
		e[i] = math.Exp(v - mx)
		sum += e[i]
	}
	for i := range e {
		e[i] /= sum
	}
	return e
}

func softmaxF64(a, b float64) []float64 {
	mx := math.Max(a, b)
	ea, eb := math.Exp(a-mx), math.Exp(b-mx)
	s := ea + eb
	return []float64{ea / s, eb / s}
}

// confidence mirrors confidence_from_probs: 1 - H(p)/ln(k).
func confidence(p []float64) float64 {
	k := len(p)
	if k < 2 {
		return 1.0
	}
	var ent float64
	for _, v := range p {
		if v < 1e-12 {
			v = 1e-12
		}
		if v > 1 {
			v = 1
		}
		ent -= v * math.Log(v)
	}
	c := 1.0 - ent/math.Log(float64(k))
	if c < 0 {
		return 0
	}
	if c > 1 {
		return 1
	}
	return c
}
