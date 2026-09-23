// Package layago: pure-Go reimplementation of the laya inference stack:
// SentencePiece-BPE tokenizer (Metaspace), laya build_sequence, ONNX Runtime
// CUDA inference and temperature/softmax postprocessing.
//
// Parity target: Python laya 0.3.5, checkpoint convaiinnovations/laya multilingual.
package layago

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const (
	repl   = "▁" // Metaspace replacement char
	unkTok = "<unk>"
)

type Tokenizer struct {
	vocab map[string]int32
	ranks map[string]int32
	unkID int32
}

// LoadTokenizer loads laya_vocab.json + laya_merges.json (dumped from tokenizer.json).
// merges must be JSON (pieces may contain raw newlines/tabs, so line formats break).
func LoadTokenizer(vocabPath, mergesPath string) (*Tokenizer, error) {
	vf, err := os.Open(vocabPath)
	if err != nil {
		return nil, err
	}
	defer vf.Close()
	var vocab map[string]int32
	if err := json.NewDecoder(vf).Decode(&vocab); err != nil {
		return nil, fmt.Errorf("vocab: %w", err)
	}
	mf, err := os.Open(mergesPath)
	if err != nil {
		return nil, err
	}
	defer mf.Close()
	var merges [][2]string
	if err := json.NewDecoder(mf).Decode(&merges); err != nil {
		return nil, fmt.Errorf("merges: %w", err)
	}
	ranks := make(map[string]int32, len(merges))
	for i, p := range merges {
		ranks[p[0]+" "+p[1]] = int32(i)
	}
	unk, ok := vocab[unkTok]
	if !ok {
		return nil, fmt.Errorf("vocab has no %s", unkTok)
	}
	return &Tokenizer{vocab: vocab, ranks: ranks, unkID: unk}, nil
}

// Encode mirrors the HF fast tokenizer pipeline:
// normalizer Replace(" " -> "▁") -> Metaspace(prepend always, split) -> BPE per piece.
func (t *Tokenizer) Encode(text string) []int32 {
	norm := strings.ReplaceAll(text, " ", repl)
	if !strings.HasPrefix(norm, repl) {
		norm = repl + norm
	}
	parts := strings.Split(norm, repl)
	out := make([]int32, 0, 64)
	for i := 1; i < len(parts); i++ {
		out = append(out, t.encodePiece(repl+parts[i])...)
	}
	return out
}

func (t *Tokenizer) encodePiece(piece string) []int32 {
	var word []string
	for _, r := range piece {
		s := string(r)
		if _, ok := t.vocab[s]; ok {
			word = append(word, s)
			continue
		}
		// byte_fallback
		for _, b := range []byte(s) {
			bs := byteTokenName(b)
			if _, ok := t.vocab[bs]; ok {
				word = append(word, bs)
			} else {
				word = append(word, unkTok)
			}
		}
	}
	for len(word) > 1 {
		bestRank := int32(-1)
		bestPair := ""
		for i := 0; i+1 < len(word); i++ {
			if r, ok := t.ranks[word[i]+" "+word[i+1]]; ok && (bestPair == "" || r < bestRank) {
				bestRank, bestPair = r, word[i]+" "+word[i+1]
			}
		}
		if bestPair == "" {
			break
		}
		a, b, _ := strings.Cut(bestPair, " ")
		merged := a + b
		nw := word[:0]
		for i := 0; i < len(word); {
			if i+1 < len(word) && word[i] == a && word[i+1] == b {
				nw = append(nw, merged)
				i += 2
			} else {
				nw = append(nw, word[i])
				i++
			}
		}
		word = nw
	}
	ids := make([]int32, len(word))
	for i, w := range word {
		if id, ok := t.vocab[w]; ok {
			ids[i] = id
		} else {
			ids[i] = t.unkID
		}
	}
	return ids
}

func byteTokenName(b byte) string {
	const hexdigits = "0123456789ABCDEF"
	return "<0x" + string(hexdigits[b>>4]) + string(hexdigits[b&0xf]) + ">"
}
