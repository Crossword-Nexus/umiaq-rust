// solver.worker.js
importScripts("../pkg/umiaq_rust.js");

let ready;
onmessage = async (e) => {
    const { type, input, wordlist, numResults } = e.data;
    if (type === "init") {
        // init wasm once
        if (!ready) {
            ready = self.init ? self.init() : (await import("./pkg/umiaq_rust.js")).default();
            await ready;
        }
        postMessage({ type: "ready" });
        return;
    }
    if (type === "solve") {
        await ready;
        try {
            const out = self.solve_equation_wasm(input, wordlist, numResults);
            postMessage({ type: "ok", results: out });
        } catch (err) {
            postMessage({ type: "err", error: String(err) });
        }
    }
};
