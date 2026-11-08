// Standalone Zig 0.17 port of the MacroStories inference (explicit allocation).
// Reads model.safetensors + tokenizer.json directly; no third-party deps.
//
// Usage:
//   zig build-exe -O ReleaseFast stories.zig
//   ./stories --root .. --seed 20261007 --max 400
//   ./stories --root .. --logits 1,9,10,11,12,16,17
const std = @import("std");

const VOCAB: usize = 378;
const HID: usize = 32;
const INTER: usize = 48;
const NH: usize = 2;
const NKV: usize = 1;
const HD: usize = 16;
const STEPS: usize = 4;
const MIX: f32 = 0.5;
const EPS: f32 = 1e-6;
const THETA: f32 = 10000.0;
const BOS: i32 = 1;
const EOS: i32 = 2;
const MAX_POS: usize = 2048;

const Weights = struct {
    embed: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    o: []const f32,
    gate: []const f32,
    up: []const f32,
    down: []const f32,
    ln1: []const f32,
    ln1b: []const f32,
    ln2: []const f32,
    ln2b: []const f32,
    norm: []const f32,
};

const KVCache = struct {
    max_seq: usize,
    k: []f32,
    v: []f32,
    scores: []f32,
};

fn dot(a: []const f32, b: []const f32, n: usize) f32 {
    const V = @Vector(8, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const av: V = a[i..][0..8].*;
        const bv: V = b[i..][0..8].*;
        acc = @mulAdd(V, av, bv, acc);
    }
    var s: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) s += a[i] * b[i];
    return s;
}

fn matvec(weight: []const f32, out_dim: usize, in_dim: usize, x: []const f32, y: []f32) void {
    var i: usize = 0;
    while (i < out_dim) : (i += 1) {
        y[i] = dot(weight[i * in_dim ..][0..in_dim], x, in_dim);
    }
}

fn rmsnorm(w: []const f32, x: []const f32, y: []f32) void {
    const ss = dot(x, x, HID);
    const inv = 1.0 / @sqrt(ss / @as(f32, HID) + EPS);
    var i: usize = 0;
    while (i < HID) : (i += 1) y[i] = w[i] * x[i] * inv;
}

fn ropeApply(vec: []f32, cosv: []const f32, sinv: []const f32) void {
    var tmp: [HD]f32 = undefined;
    var d: usize = 0;
    while (d < HD) : (d += 1) {
        const rot = if (d < HD / 2) -vec[d + HD / 2] else vec[d - HD / 2];
        tmp[d] = vec[d] * cosv[d] + rot * sinv[d];
    }
    @memcpy(vec[0..HD], tmp[0..HD]);
}

fn forwardToken(
    w: *const Weights,
    token: i32,
    pos: usize,
    cache: *KVCache,
    invfreq: []const f32,
    logits: []f32,
) void {
    var x: [HID]f32 = undefined;
    const ti: usize = @intCast(token);
    var i: usize = 0;
    while (i < HID) : (i += 1) x[i] = w.embed[ti * HID + i];

    var cosv: [HD]f32 = undefined;
    var sinv: [HD]f32 = undefined;
    var j: usize = 0;
    while (j < HD / 2) : (j += 1) {
        const a = @as(f32, @floatFromInt(pos)) * invfreq[j];
        const c = @cos(a);
        const s = @sin(a);
        cosv[j] = c;
        cosv[j + HD / 2] = c;
        sinv[j] = s;
        sinv[j + HD / 2] = s;
    }
    const scale = 1.0 / @sqrt(@as(f32, HD));

    var r: [HID]f32 = undefined;
    var cur: [HID]f32 = undefined;
    var qv: [NH * HD]f32 = undefined;
    var kv: [NKV * HD]f32 = undefined;
    var vv: [NKV * HD]f32 = undefined;
    var first_v: [NKV * HD]f32 = undefined;
    var attn: [NH * HD]f32 = undefined;
    var out: [HID]f32 = undefined;
    var tmp: [HID]f32 = undefined;
    var z: [HID]f32 = undefined;
    var g: [INTER]f32 = undefined;
    var u: [INTER]f32 = undefined;
    var m: [INTER]f32 = undefined;

    var step: usize = 0;
    while (step < STEPS) : (step += 1) {
        @memcpy(r[0..HID], x[0..HID]);
        rmsnorm(w.ln1, x[0..HID], cur[0..HID]);
        matvec(w.q, NH * HD, HID, cur[0..HID], qv[0..NH * HD]);
        matvec(w.k, NKV * HD, HID, cur[0..HID], kv[0..NKV * HD]);
        matvec(w.v, NKV * HD, HID, cur[0..HID], vv[0..NKV * HD]);
        if (step == 0) {
            @memcpy(first_v[0 .. NKV * HD], vv[0 .. NKV * HD]);
        } else {
            var t: usize = 0;
            while (t < NKV * HD) : (t += 1) vv[t] = (1.0 - MIX) * vv[t] + MIX * first_v[t];
        }
        var head: usize = 0;
        while (head < NH) : (head += 1) ropeApply(qv[head * HD ..][0..HD], cosv[0..HD], sinv[0..HD]);
        ropeApply(kv[0 .. NKV * HD], cosv[0..HD], sinv[0..HD]);

        const koff = (step * cache.max_seq + pos) * NKV * HD;
        @memcpy(cache.k[koff..][0 .. NKV * HD], kv[0 .. NKV * HD]);
        @memcpy(cache.v[koff..][0 .. NKV * HD], vv[0 .. NKV * HD]);

        head = 0;
        while (head < NH) : (head += 1) {
            const qh = qv[head * HD ..][0..HD];
            var mx: f32 = -1e30;
            var s2: usize = 0;
            while (s2 <= pos) : (s2 += 1) {
                const kk = cache.k[(step * cache.max_seq + s2) * NKV * HD ..][0 .. NKV * HD];
                const acc = dot(qh, kk, HD) * scale;
                cache.scores[s2] = acc;
                if (acc > mx) mx = acc;
            }
            var sum: f32 = 0;
            s2 = 0;
            while (s2 <= pos) : (s2 += 1) {
                const e = @exp(cache.scores[s2] - mx);
                cache.scores[s2] = e;
                sum += e;
            }
            const inv = 1.0 / sum;
            const ao = attn[head * HD ..][0..HD];
            var d: usize = 0;
            while (d < HD) : (d += 1) ao[d] = 0;
            s2 = 0;
            while (s2 <= pos) : (s2 += 1) {
                const p = cache.scores[s2] * inv;
                const vs = cache.v[(step * cache.max_seq + s2) * NKV * HD ..][0 .. NKV * HD];
                d = 0;
                while (d < HD) : (d += 1) ao[d] += p * vs[d];
            }
        }

        matvec(w.o, HID, NH * HD, attn[0 .. NH * HD], out[0..HID]);
        rmsnorm(w.ln1b, out[0..HID], tmp[0..HID]);
        i = 0;
        while (i < HID) : (i += 1) x[i] = r[i] + tmp[i];

        @memcpy(r[0..HID], x[0..HID]);
        rmsnorm(w.ln2, x[0..HID], z[0..HID]);
        matvec(w.gate, INTER, HID, z[0..HID], g[0..INTER]);
        matvec(w.up, INTER, HID, z[0..HID], u[0..INTER]);
        i = 0;
        while (i < INTER) : (i += 1) {
            const v = g[i];
            m[i] = (v / (1.0 + @exp(-v))) * u[i];
        }
        matvec(w.down, HID, INTER, m[0..INTER], out[0..HID]);
        rmsnorm(w.ln2b, out[0..HID], tmp[0..HID]);
        i = 0;
        while (i < HID) : (i += 1) x[i] = r[i] + tmp[i];

        rmsnorm(w.norm, x[0..HID], tmp[0..HID]);
        @memcpy(x[0..HID], tmp[0..HID]);
    }

    i = 0;
    while (i < VOCAB) : (i += 1) logits[i] = dot(w.embed[i * HID ..][0..HID], x[0..HID], HID);
}

fn lessThanDesc(probs: []const f32, a: usize, b: usize) bool {
    return probs[a] > probs[b];
}

fn sample(logits: []const f32, temp: f32, top_p: f32, rng: std.Random, probs: []f32, order: []usize) i32 {
    var mx: f32 = -1e30;
    var i: usize = 0;
    while (i < VOCAB) : (i += 1) {
        const v = logits[i] / temp;
        probs[i] = v;
        if (v > mx) mx = v;
    }
    var sum: f32 = 0;
    i = 0;
    while (i < VOCAB) : (i += 1) {
        probs[i] = @exp(probs[i] - mx);
        sum += probs[i];
    }
    i = 0;
    while (i < VOCAB) : (i += 1) probs[i] /= sum;

    i = 0;
    while (i < VOCAB) : (i += 1) order[i] = i;
    // PDQ (pattern-defeating quicksort); std.sort.heap was ~4x slower here.
    std.sort.pdq(usize, order, probs, lessThanDesc);

    var cum: f32 = 0;
    var keep: usize = 0;
    while (keep < VOCAB) : (keep += 1) {
        cum += probs[order[keep]];
        if (cum >= top_p) {
            keep += 1;
            break;
        }
    }
    if (keep == 0) keep = 1;

    var norm: f32 = 0;
    i = 0;
    while (i < keep) : (i += 1) norm += probs[order[i]];
    const r = rng.float(f32) * norm;
    var acc: f32 = 0;
    i = 0;
    while (i < keep) : (i += 1) {
        acc += probs[order[i]];
        if (r <= acc) return @intCast(order[i]);
    }
    return @intCast(order[keep - 1]);
}

fn buildByteDecoder(map: *[512]?u8) void {
    var bs: [256]u16 = undefined;
    var cs: [256]u16 = undefined;
    var m: usize = 0;
    var b: u16 = 33;
    while (b <= 126) : (b += 1) {
        bs[m] = b;
        cs[m] = b;
        m += 1;
    }
    b = 161;
    while (b <= 172) : (b += 1) {
        bs[m] = b;
        cs[m] = b;
        m += 1;
    }
    b = 174;
    while (b <= 255) : (b += 1) {
        bs[m] = b;
        cs[m] = b;
        m += 1;
    }
    var n: u16 = 0;
    b = 0;
    while (b < 256) : (b += 1) {
        var present = false;
        for (bs[0..m]) |x| {
            if (x == b) {
                present = true;
                break;
            }
        }
        if (!present) {
            bs[m] = b;
            cs[m] = 256 + n;
            n += 1;
            m += 1;
        }
    }
    var i: usize = 0;
    while (i < m) : (i += 1) map[cs[i]] = @intCast(bs[i]);
}

fn appendByte(out: []u8, len: *usize, byte: u8) void {
    if (len.* < out.len) {
        out[len.*] = byte;
        len.* += 1;
    }
}

fn appendSlice(out: []u8, len: *usize, bytes: []const u8) void {
    for (bytes) |b| appendByte(out, len, b);
}

fn decodeTokenBytes(tok: []const u8, map: *const [512]?u8, out: []u8, len: *usize) void {
    var i: usize = 0;
    while (i < tok.len) {
        const c = tok[i];
        var cp: u32 = 0;
        var n: usize = 1;
        if (c < 0x80) {
            cp = c;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            n = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            n = 3;
        } else {
            cp = c & 0x07;
            n = 4;
        }
        var j: usize = 1;
        while (j < n and i + j < tok.len) : (j += 1) cp = (cp << 6) | (tok[i + j] & 0x3F);
        if (cp < map.len) {
            if (map[cp]) |byte| appendByte(out, len, byte);
        }
        i += n;
    }
}

fn detokenize(ids: []const i32, id2tok: *const [VOCAB]?[]const u8, map: *const [512]?u8, out: []u8) []const u8 {
    var len: usize = 0;
    for (ids) |id| {
        switch (id) {
            0, BOS, EOS, 3 => continue,
            4 => {
                appendSlice(out, &len, "Alex");
                continue;
            },
            5 => {
                appendSlice(out, &len, "Robin");
                continue;
            },
            6 => {
                appendSlice(out, &len, "Casey");
                continue;
            },
            else => {},
        }
        const idx: usize = @intCast(id);
        if (idx < VOCAB) {
            if (id2tok[idx]) |tok| decodeTokenBytes(tok, map, out, &len);
        }
    }
    return std.mem.trim(u8, out[0..len], " \t\n\r");
}

fn loadTensor(
    obj: std.json.ObjectMap,
    name: []const u8,
    data_base: usize,
    file_bytes: []const u8,
    expected: usize,
    alloc: std.mem.Allocator,
) ![]f32 {
    const tv = obj.get(name) orelse return error.MissingTensor;
    const to = tv.object;
    const dtype = (to.get("dtype") orelse return error.MissingDtype).string;
    if (!std.mem.eql(u8, dtype, "F32")) return error.NotF32;
    const offs = (to.get("data_offsets") orelse return error.MissingOffsets).array.items;
    const start: usize = @intCast(offs[0].integer);
    const end: usize = @intCast(offs[1].integer);
    const numel = (end - start) / 4;
    if (numel != expected) return error.BadShape;
    const dst = try alloc.alloc(f32, numel);
    @memcpy(std.mem.sliceAsBytes(dst), file_bytes[data_base + start .. data_base + end]);
    return dst;
}

fn loadWeights(
    header: []const u8,
    file_bytes: []const u8,
    data_base: usize,
    alloc: std.mem.Allocator,
) !Weights {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, alloc, header, .{});
    const obj = root.object;
    const p = "model.layers.0.";
    return .{
        .embed = try loadTensor(obj, "model.embed_tokens.weight", data_base, file_bytes, VOCAB * HID, alloc),
        .q = try loadTensor(obj, p ++ "self_attn.q_proj.weight", data_base, file_bytes, NH * HD * HID, alloc),
        .k = try loadTensor(obj, p ++ "self_attn.k_proj.weight", data_base, file_bytes, NKV * HD * HID, alloc),
        .v = try loadTensor(obj, p ++ "self_attn.v_proj.weight", data_base, file_bytes, NKV * HD * HID, alloc),
        .o = try loadTensor(obj, p ++ "self_attn.o_proj.weight", data_base, file_bytes, HID * NH * HD, alloc),
        .gate = try loadTensor(obj, p ++ "mlp.gate_proj.weight", data_base, file_bytes, INTER * HID, alloc),
        .up = try loadTensor(obj, p ++ "mlp.up_proj.weight", data_base, file_bytes, INTER * HID, alloc),
        .down = try loadTensor(obj, p ++ "mlp.down_proj.weight", data_base, file_bytes, HID * INTER, alloc),
        .ln1 = try loadTensor(obj, p ++ "input_layernorm.weight", data_base, file_bytes, HID, alloc),
        .ln1b = try loadTensor(obj, p ++ "input_layernorm_2.weight", data_base, file_bytes, HID, alloc),
        .ln2 = try loadTensor(obj, p ++ "post_attention_layernorm.weight", data_base, file_bytes, HID, alloc),
        .ln2b = try loadTensor(obj, p ++ "post_attention_layernorm_2.weight", data_base, file_bytes, HID, alloc),
        .norm = try loadTensor(obj, "model.norm.weight", data_base, file_bytes, HID, alloc),
    };
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    // Explicit arena: every model/workspace allocation goes through `arena`,
    // and is released in one place at the end (defer).
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var root: []const u8 = ".";
    var seed: u64 = 20261007;
    var max_new: usize = 512;
    var temp: f32 = 0.5;
    var top_p: f32 = 0.9;
    var dump_ids: [MAX_POS]i32 = undefined;
    var n_dump: usize = 0;
    var quiet = false;

    var args_it = std.process.Args.Iterator.init(init.minimal.args);
    _ = args_it.next();
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--root")) {
            root = args_it.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            seed = try std.fmt.parseInt(u64, args_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--max")) {
            max_new = try std.fmt.parseInt(usize, args_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--temp")) {
            temp = try std.fmt.parseFloat(f32, args_it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            top_p = try std.fmt.parseFloat(f32, args_it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--quiet")) {
            quiet = true;
        } else if (std.mem.eql(u8, arg, "--logits")) {
            const s = args_it.next() orelse return error.MissingValue;
            var parts = std.mem.splitScalar(u8, s, ',');
            while (parts.next()) |part| {
                if (part.len == 0) continue;
                if (n_dump >= MAX_POS) return error.TooManyIds;
                dump_ids[n_dump] = try std.fmt.parseInt(i32, part, 10);
                n_dump += 1;
            }
        }
    }

    if (temp <= 0) return error.BadTemperature;
    if (top_p <= 0 or top_p > 1) return error.BadTopP;
    if (max_new < 1 or max_new > MAX_POS) return error.BadMaxTokens;
    for (dump_ids[0..n_dump]) |id| {
        if (id < 0 or id >= VOCAB) return error.TokenOutOfRange;
    }

    const model_path = try std.fmt.allocPrint(arena, "{s}/model.safetensors", .{root});
    const tok_path = try std.fmt.allocPrint(arena, "{s}/tokenizer.json", .{root});
    const cwd = std.Io.Dir.cwd();
    const model_bytes = try cwd.readFileAlloc(io, model_path, arena, .limited(1 << 24));
    const tok_bytes = try cwd.readFileAlloc(io, tok_path, arena, .limited(1 << 24));

    var header_len: u64 = 0;
    var bi: usize = 0;
    while (bi < 8) : (bi += 1) header_len |= @as(u64, model_bytes[bi]) << @intCast(8 * bi);
    const hlen: usize = @intCast(header_len);
    const header = model_bytes[8 .. 8 + hlen];
    const data_base = 8 + hlen;
    const w = try loadWeights(header, model_bytes, data_base, arena);

    const tok_json = try std.json.parseFromSliceLeaky(std.json.Value, arena, tok_bytes, .{});
    const tok_model = (tok_json.object.get("model") orelse return error.MissingModel).object;
    const vocab = (tok_model.get("vocab") orelse return error.MissingVocab).object;
    var id2tok: [VOCAB]?[]const u8 = undefined;
    for (&id2tok) |*p| p.* = null;
    var vit = vocab.iterator();
    while (vit.next()) |e| {
        const id: usize = @intCast(e.value_ptr.*.integer);
        if (id < VOCAB) id2tok[id] = e.key_ptr.*;
    }
    var cp2byte: [512]?u8 = undefined;
    for (&cp2byte) |*p| p.* = null;
    buildByteDecoder(&cp2byte);

    var invfreq: [HD / 2]f32 = undefined;
    var j: usize = 0;
    while (j < HD / 2) : (j += 1) {
        const e = -(2.0 * @as(f32, @floatFromInt(j))) / @as(f32, HD);
        invfreq[j] = @exp(e * @log(THETA));
    }

    var out_buf: [4096]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &out_buf);
    const writer = &fw.interface;

    if (n_dump > 0) {
        const seq_len = n_dump + 2;
        var cache = KVCache{
            .max_seq = seq_len,
            .k = try arena.alloc(f32, STEPS * seq_len * NKV * HD),
            .v = try arena.alloc(f32, STEPS * seq_len * NKV * HD),
            .scores = try arena.alloc(f32, seq_len),
        };
        var logits: [VOCAB]f32 = undefined;
        var pos: usize = 0;
        while (pos < n_dump) : (pos += 1) forwardToken(&w, dump_ids[pos], pos, &cache, invfreq[0..], logits[0..]);
        var i: usize = 0;
        while (i < VOCAB) : (i += 1) {
            try writer.print("{d:.6}", .{logits[i]});
            try writer.writeAll(if (i + 1 == VOCAB) "\n" else " ");
        }
        try writer.flush();
        return;
    }

    const max_seq = max_new + 2;
    var cache = KVCache{
        .max_seq = max_seq,
        .k = try arena.alloc(f32, STEPS * max_seq * NKV * HD),
        .v = try arena.alloc(f32, STEPS * max_seq * NKV * HD),
        .scores = try arena.alloc(f32, max_seq),
    };

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    var probs: [VOCAB]f32 = undefined;
    var order: [VOCAB]usize = undefined;
    var logits: [VOCAB]f32 = undefined;

    const ids = try arena.alloc(i32, max_seq);
    var n_ids: usize = 0;
    var cur: i32 = BOS;
    var pos: usize = 0;
    while (pos < max_new) : (pos += 1) {
        forwardToken(&w, cur, pos, &cache, invfreq[0..], logits[0..]);
        const nxt = sample(logits[0..], temp, top_p, rng, probs[0..], order[0..]);
        ids[n_ids] = nxt;
        n_ids += 1;
        cur = nxt;
        if (nxt == EOS) break;
    }

    const text_buf = try arena.alloc(u8, 1 << 16);
    const story = detokenize(ids[0..n_ids], &id2tok, &cp2byte, text_buf);
    if (!quiet) {
        try writer.print("{s}\n", .{story});
        try writer.flush();
    }
    std.debug.print("[tokens {d}]\n", .{n_ids});
}
