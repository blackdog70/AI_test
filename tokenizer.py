"""
Tokenizzatore didattico - Byte Pair Encoding (BPE)
===================================================
Implementa l'algoritmo BPE usato in produzione da GPT-2/GPT-4, LLaMA, ecc.

Concetti chiave fedeli alla ricerca:
  - Vocabolario costruito bottom-up partendo dai byte grezzi (256 token base)
  - Merge iterativo delle coppie piu' frequenti (Sennrich et al., 2016)
  - Token ID numerici interi (come nei transformer reali)
  - Testo -> token IDs -> testo (encode/decode simmetrici)
  - Special token: <|endoftext|>, <|pad|>, ecc.
"""

import re
import json
from collections import defaultdict
from typing import Optional


def get_pairs(ids):
    counts = defaultdict(int)
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def merge(ids, pair, new_id):
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


class BPETokenizer:
    """
    Tokenizzatore Byte Pair Encoding (BPE) didattico.

    Vocabolario iniziale: 256 byte (ogni byte e' un token da 0 a 255).
    Vocabolario finale:   256 + num_merges token.

    special_tokens: token con ID riservati aggiunti dopo il vocabolario BPE.
    """

    # Pattern GPT-2 (richiede la libreria `regex`; fallback a `re` standard)
    _GPT2_SPLIT_PATTERN_STR = (
        r"'(?:[sdmt]|ll|ve|re)|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}{1,3}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    def __init__(self, special_tokens=None):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.special_tokens = special_tokens or {}
        self._build_decode_table()

    def train(self, text, vocab_size, verbose=True):
        """Addestra il tokenizzatore. vocab_size >= 256."""
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        chunks = self._pretokenize(text)
        ids_per_chunk = [list(chunk.encode("utf-8")) for chunk in chunks]

        if verbose:
            total = sum(len(c) for c in ids_per_chunk)
            print(f"[train] {len(text)} caratteri -> {total} byte-token iniziali")
            print(f"[train] vocabolario target: {vocab_size} ({num_merges} merge)\n")

        for step in range(num_merges):
            counts = defaultdict(int)
            for ids in ids_per_chunk:
                for pair, cnt in get_pairs(ids).items():
                    counts[pair] += cnt
            if not counts:
                break

            best = max(counts, key=lambda p: counts[p])
            new_id = 256 + step
            ids_per_chunk = [merge(ids, best, new_id) for ids in ids_per_chunk]
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            if verbose:
                try:
                    readable = self.vocab[new_id].decode("utf-8")
                except UnicodeDecodeError:
                    readable = repr(self.vocab[new_id])
                print(f"  merge {step+1:4d}/{num_merges}: "
                      f"({best[0]:4d},{best[1]:4d}) -> {new_id:4d}  "
                      f"freq={counts[best]:6d}  token={readable!r}")

        for tok_str, tok_id in self.special_tokens.items():
            self.vocab[tok_id] = tok_str.encode("utf-8")
        self._build_decode_table()
        if verbose:
            print(f"\n[train] completato. Vocabolario: {len(self.vocab)} token.")

    def encode(self, text):
        """Testo -> lista di token ID (interi)."""
        if self.special_tokens:
            return self._encode_with_special(text)
        ids = []
        for chunk in self._pretokenize(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids):
        """Lista di token ID -> testo."""
        raw = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return raw.decode("utf-8", errors="replace")

    def token_info(self, token_id):
        """Informazioni su un singolo token (utile per debug didattico)."""
        raw = self.vocab.get(token_id)
        if raw is None:
            return {"id": token_id, "error": "non trovato"}
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        return {
            "id": token_id,
            "bytes": list(raw),
            "hex": raw.hex(),
            "text": text,
            "is_special": token_id in set(self.special_tokens.values()),
            "is_base_byte": token_id < 256,
            "is_merged": 256 <= token_id < 256 + len(self.merges),
        }

    def show_tokenization(self, text):
        """Visualizzazione colorata della tokenizzazione nel terminale."""
        ids = self.encode(text)
        COLORS = ["\033[41m", "\033[42m", "\033[43m", "\033[44m", "\033[45m", "\033[46m"]
        RESET = "\033[0m"
        print(f"\nTesto: {text!r}")
        print(f"Token IDs ({len(ids)} token): {ids}\n")
        colored = ""
        for i, tid in enumerate(ids):
            raw = self.vocab.get(tid, b"?")
            try:
                t = raw.decode("utf-8")
            except UnicodeDecodeError:
                t = f"[{raw.hex()}]"
            colored += f"{COLORS[i % len(COLORS)]}{t}{RESET}"
        print(colored + "\n")

    def vocab_summary(self):
        n_base = sum(1 for i in self.vocab if i < 256)
        n_merged = sum(1 for i in self.vocab if 256 <= i < 256 + len(self.merges))
        print(f"Vocabolario: {len(self.vocab)} token totali")
        print(f"  - {n_base} token base (byte 0-255)")
        print(f"  - {n_merged} token fusi (BPE merge)")
        print(f"  - {len(self.special_tokens)} special token: {list(self.special_tokens.keys())}")

    def save(self, path):
        data = {
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[save] modello salvato in {path!r}")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(special_tokens=data.get("special_tokens", {}))
        for a, b, c in data["merges"]:
            tok.merges[(a, b)] = c
            tok.vocab[c] = tok.vocab[a] + tok.vocab[b]
        tok._build_decode_table()
        print(f"[load] {len(tok.merges)} merge caricati da {path!r}")
        return tok

    def _pretokenize(self, text):
        try:
            import regex
            return regex.compile(self._GPT2_SPLIT_PATTERN_STR, regex.UNICODE).findall(text)
        except ImportError:
            return re.compile(r"'(?:[sdmt]|ll|ve|re)|\d{1,3}|[^\s\w]+|\w+|\s+", re.UNICODE).findall(text)

    def _encode_chunk(self, chunk_bytes):
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            pairs = get_pairs(ids)
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = merge(ids, best, self.merges[best])
        return ids

    def _encode_with_special(self, text):
        pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        ids = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                for chunk in self._pretokenize(part):
                    ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def _build_decode_table(self):
        self._special_by_id = {v: k for k, v in self.special_tokens.items()}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

TESTO_ESEMPIO = """
L'intelligenza artificiale (IA) e' una branca dell'informatica che studia come
creare sistemi in grado di eseguire compiti che normalmente richiedono
intelligenza umana. Tra questi: riconoscimento vocale, visione artificiale,
traduzione automatica e ragionamento logico.

I modelli linguistici di grandi dimensioni (LLM) come GPT-4 o LLaMA utilizzano
architetture transformer con miliardi di parametri. Il tokenizzatore e' il primo
componente della pipeline: converte il testo grezzo in una sequenza di interi
(token ID) che la rete neurale puo' elaborare.

hello world! 123 test... tokenization is fun :)
""".strip()


def demo():
    print("=" * 65)
    print("  Tokenizzatore BPE didattico  --  ispirato a GPT-2 / tiktoken")
    print("=" * 65)

    special = {"<|endoftext|>": 300, "<|pad|>": 301, "<|unk|>": 302}
    tok = BPETokenizer(special_tokens=special)

    print("\n[1] TRAINING...")
    tok.train(TESTO_ESEMPIO, vocab_size=300, verbose=True)

    print("\n" + "=" * 65)
    tok.vocab_summary()

    print("\n[2] ENCODE / DECODE")
    print("-" * 65)
    for frase in ["intelligenza artificiale", "tokenization is fun", "LLM <|endoftext|>", "123 parametri"]:
        ids = tok.encode(frase)
        decoded = tok.decode(ids)
        ok = "OK" if decoded == frase else "ERRORE"
        print(f"  {frase!r}")
        print(f"  -> IDs: {ids}")
        print(f"  -> decode: {decoded!r}  [{ok}]\n")

    print("[3] VISUALIZZAZIONE")
    tok.show_tokenization("L'intelligenza artificiale")
    tok.show_tokenization("hello world! 123")

    print("[4] ISPEZIONE TOKEN")
    print("-" * 65)
    for tid in [65, 256, 257, 300]:
        print(f"  {tok.token_info(tid)}")

    print("\n[5] COMPRESSIONE")
    print("-" * 65)
    for frase in ["intelligenza artificiale", "tokenization"]:
        nb = len(frase.encode("utf-8"))
        nt = len(tok.encode(frase))
        print(f"  {frase!r}: {nb} byte -> {nt} token (ratio {nb/nt:.2f}x)")

    print("\n[6] SALVATAGGIO / RELOAD")
    print("-" * 65)
    tok.save("tokenizer_model.json")
    tok2 = BPETokenizer.load("tokenizer_model.json")
    frase_test = "intelligenza artificiale"
    assert tok.encode(frase_test) == tok2.encode(frase_test)
    print("  Verifica encode dopo reload: OK")

    print("\n" + "=" * 65)
    print("  Demo completata.")
    print("=" * 65)


if __name__ == "__main__":
    demo()
