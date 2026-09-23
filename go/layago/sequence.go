package layago

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
)

// Cfg mirrors laya_go_cfg.json (dumped from rl_agent_config.json + tokenizer).
type Cfg struct {
	MaxLen               int                `json:"max_len"`
	HeadMaxLen           int                `json:"head_max_len"`
	Temperature          []float64          `json:"temperature"`
	TemperatureByOptions map[string]float64 `json:"temperature_by_options"`
	CLSTokenID           int32              `json:"cls_token_id"`
	SEPTokenID           int32              `json:"sep_token_id"`
	PadTokenID           int32              `json:"pad_token_id"`
	MaskTokenID          int32              `json:"mask_token_id"`
	MaskToken            string             `json:"mask_token"`
	UnkTokenID           int32              `json:"unk_token_id"`
	QTypes               map[string]int32   `json:"qtypes"`
}

func LoadCfg(path string) (*Cfg, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Cfg
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, err
	}
	if c.TemperatureByOptions == nil {
		c.TemperatureByOptions = map[string]float64{}
	}
	return &c, nil
}

// QInternal mirrors laya _to_internal.
type QInternal struct {
	T        string
	Ins      string
	Crit     *Obj  // choice / noul criteria (noul may be nil)
	CritList []any // score levels, in order
}

func ToInternal(qdef *Obj) (*QInternal, error) {
	tAny, _ := qdef.Get("type")
	t, ok := tAny.(string)
	if !ok {
		return nil, fmt.Errorf("question missing type")
	}
	crit := qdef.M["criteria"]
	if t == "choice" {
		if arr, ok := crit.([]any); ok {
			o := &Obj{M: map[string]any{}}
			for _, c := range arr {
				o.Keys = append(o.Keys, fmt.Sprintf("%v", c))
				o.M[fmt.Sprintf("%v", c)] = nil
			}
			crit = o
		}
	}
	ins := qdef.M["instructions"]
	var insStr string
	switch v := ins.(type) {
	case string:
		insStr = v
	case nil:
		return nil, fmt.Errorf("question missing instructions")
	default:
		insStr = pyDumps(v)
	}
	q := &QInternal{T: t, Ins: insStr}
	switch c := crit.(type) {
	case *Obj:
		q.Crit = c
	case []any:
		q.CritList = c
	}
	if t == "score" && len(q.CritList) == 0 {
		return nil, fmt.Errorf("score question needs a criteria list")
	}
	return q, nil
}

// serializeState mirrors laya.common.serialize_state (Python json.dumps, no ascii).
func serializeState(state any) string {
	if s, ok := state.(string); ok {
		return s
	}
	return pyDumps(state)
}

// renderCriterion mirrors laya.common.render_criterion.
func renderCriterion(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return pyDumps(v)
}

// RenderOptions mirrors laya.common.render_options (label-index order).
func RenderOptions(q *QInternal) []string {
	switch q.T {
	case "choice":
		out := make([]string, 0, len(q.Crit.Keys))
		for _, k := range q.Crit.Keys {
			v := q.Crit.M[k]
			if v == nil {
				out = append(out, k)
				continue
			}
			if s, ok := v.(string); ok && s == "" {
				out = append(out, k)
				continue
			}
			out = append(out, k+": "+renderCriterion(v))
		}
		return out
	case "score":
		out := make([]string, len(q.CritList))
		for i, c := range q.CritList {
			out[i] = fmt.Sprintf("level %d: %s", i, renderCriterion(c))
		}
		return out
	case "noul":
		var fc, tc any
		if q.Crit != nil {
			fc = q.Crit.M["false"]
			tc = q.Crit.M["true"]
		}
		fs := "no, the statement does not hold"
		if fc != nil && fc != "" {
			fs = renderCriterion(fc)
		}
		ts := "yes, the statement holds"
		if tc != nil && tc != "" {
			ts = renderCriterion(tc)
		}
		return []string{"false: " + fs, "true: " + ts}
	}
	return nil
}

// ReplaceMask mirrors the mask-token scrubbing done before tokenizing.
func ReplaceMask(s, mask string) string {
	if mask == "" {
		return s
	}
	return strings.ReplaceAll(s, mask, " ")
}

// PyDumps exposes pyDumps for callers.
func PyDumps(v any) string { return pyDumps(v) }

// Item is one built sequence ready for the model.
type Item struct {
	IDs     []int32
	Markers []int32
	QType   int32
}

// BuildSequence mirrors laya.common.build_sequence (truncate_left=false path).
func BuildSequence(tok *Tokenizer, cfg *Cfg, state any, q *QInternal) (*Item, error) {
	opts := RenderOptions(q)
	maskTok := cfg.MaskToken

	ins := strings.ReplaceAll(q.Ins, maskTok, " ")
	headIDs := tok.Encode(q.T + " question: " + ins)

	optIDs := make([][]int32, len(opts))
	sum := 0
	for i, o := range opts {
		ids := tok.Encode(" " + strings.ReplaceAll(o, maskTok, " "))
		if len(ids) > 48 {
			ids = ids[:48]
		}
		optIDs[i] = ids
		sum += len(ids)
	}
	optBudget := cfg.HeadMaxLen - sum
	if optBudget < 16 {
		per := (cfg.HeadMaxLen - 16) / max(1, len(optIDs))
		if per < 4 {
			per = 4
		}
		for i := range optIDs {
			if len(optIDs[i]) > per {
				optIDs[i] = optIDs[i][:per]
			}
		}
		optBudget = cfg.HeadMaxLen
		for _, o := range optIDs {
			optBudget -= len(o)
		}
	}
	hl := len(headIDs)
	if hl > optBudget {
		hl = optBudget
	}
	if hl < 8 {
		hl = 8
	}

	ids := make([]int32, 0, cfg.MaxLen)
	ids = append(ids, cfg.CLSTokenID)
	ids = append(ids, headIDs[:hl]...)
	ids = append(ids, cfg.SEPTokenID)
	var markers []int32
	for _, o := range optIDs {
		markers = append(markers, int32(len(ids)))
		ids = append(ids, cfg.MaskTokenID)
		ids = append(ids, o...)
	}
	ids = append(ids, cfg.SEPTokenID)

	room := cfg.MaxLen - len(ids) - 1
	if room < 0 {
		room = 0
	}
	stText := strings.ReplaceAll(serializeState(state), maskTok, " ")
	stIDs := tok.Encode(stText)
	if len(stIDs) > room {
		stIDs = stIDs[:room]
	}
	ids = append(ids, stIDs...)
	ids = append(ids, cfg.SEPTokenID)
	if len(ids) > cfg.MaxLen {
		ids = ids[:cfg.MaxLen]
	}
	mk := markers[:0]
	for _, m := range markers {
		if int(m) < cfg.MaxLen {
			mk = append(mk, m)
		}
	}
	return &Item{IDs: ids, Markers: mk, QType: cfg.QTypes[q.T]}, nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// TempScale mirrors agent temperature lookup (all-1.0 for this checkpoint).
func (c *Cfg) TempScale(qtype int32, k int) float64 {
	bucket := tempBucket(qtype, k)
	if v, ok := c.TemperatureByOptions[bucket]; ok {
		return clampTemp(v)
	}
	if int(qtype) < len(c.Temperature) {
		return clampTemp(c.Temperature[qtype])
	}
	return 1.0
}

func tempBucket(qtype int32, k int) string {
	var name string
	switch qtype {
	case 0:
		name = "choice"
	case 1:
		name = "score"
	default:
		name = "noul"
	}
	size := "11+"
	if k <= 2 {
		size = "2"
	} else if k <= 5 {
		size = "3-5"
	} else if k <= 10 {
		size = "6-10"
	}
	return name + ":" + size
}

func clampTemp(t float64) float64 {
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return 1.0
	}
	return math.Min(5.0, math.Max(0.5, t))
}
