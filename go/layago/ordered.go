package layago

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// Obj is a JSON object that preserves key order (Python dict semantics).
type Obj struct {
	Keys []string
	M    map[string]any
}

func (o *Obj) Get(k string) (any, bool) {
	v, ok := o.M[k]
	return v, ok
}

func (o *Obj) Set(k string, v any) {
	if _, ok := o.M[k]; !ok {
		o.Keys = append(o.Keys, k)
	}
	if o.M == nil {
		o.M = map[string]any{}
	}
	o.M[k] = v
}

func (o *Obj) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, k := range o.Keys {
		if i > 0 {
			b.WriteString(", ")
		}
		kb, _ := json.Marshal(k)
		b.Write(kb)
		b.WriteString(": ")
		vb, err := json.Marshal(o.M[k])
		if err != nil {
			return nil, err
		}
		b.Write(vb)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// decodeValue decodes JSON preserving object key order everywhere.
func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			o := &Obj{M: map[string]any{}}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, ok := kt.(string)
				if !ok {
					return nil, fmt.Errorf("bad key %v", kt)
				}
				v, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				o.Keys = append(o.Keys, key)
				o.M[key] = v
			}
			if _, err := dec.Token(); err != nil { // consume '}'
				return nil, err
			}
			return o, nil
		case '[':
			var arr []any
			for dec.More() {
				v, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				arr = append(arr, v)
			}
			if _, err := dec.Token(); err != nil { // consume ']'
				return nil, err
			}
			if arr == nil {
				arr = []any{}
			}
			return arr, nil
		}
		return nil, fmt.Errorf("unexpected delim %v", t)
	default:
		return tok, nil
	}
}

// DecodeOrdered reads one JSON value (object/array/scalar) preserving order.
func DecodeOrdered(r io.Reader) (any, error) {
	dec := json.NewDecoder(r)
	dec.UseNumber()
	return decodeValue(dec)
}

// pyDumps mirrors Python json.dumps(x, ensure_ascii=False):
// separators (', ', ': '), raw unicode, Python string escaping.
func pyDumps(v any) string {
	var b strings.Builder
	pyDump(&b, v)
	return b.String()
}

func pyDump(b *strings.Builder, v any) {
	switch x := v.(type) {
	case *Obj:
		b.WriteByte('{')
		for i, k := range x.Keys {
			if i > 0 {
				b.WriteString(", ")
			}
			pyString(b, k)
			b.WriteString(": ")
			pyDump(b, x.M[k])
		}
		b.WriteByte('}')
	case []any:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteString(", ")
			}
			pyDump(b, e)
		}
		b.WriteByte(']')
	case string:
		pyString(b, x)
	case bool:
		if x {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case nil:
		b.WriteString("null")
	case json.Number:
		b.WriteString(x.String())
	case float64:
		b.WriteString(pyFloat(x))
	case int:
		b.WriteString(fmt.Sprintf("%d", x))
	default:
		pyString(b, fmt.Sprintf("%v", x))
	}
}

func pyFloat(f float64) string {
	// Python repr of floats: shortest round-trip, always with .0 for integers.
	s := fmt.Sprintf("%v", f)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

const hexDigits = "0123456789abcdef"

func pyString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString("\\\"")
		case '\\':
			b.WriteString("\\\\")
		case '\n':
			b.WriteString("\\n")
		case '\r':
			b.WriteString("\\r")
		case '\t':
			b.WriteString("\\t")
		default:
			if r < 0x20 {
				b.WriteString("\\u00")
				b.WriteByte(hexDigits[r>>4])
				b.WriteByte(hexDigits[r&0xf])
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}
