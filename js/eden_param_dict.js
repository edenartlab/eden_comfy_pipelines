import { app } from "../../scripts/app.js";

// Frontend for Eden_Save_Param_Dict: each row is a `value_N` socket plus a `key_N` text widget.
// Connecting the last socket adds a new empty row; trailing empty rows are trimmed back to one.
// The backend reads value_N as kwargs and key_N from the prompt (neither is declared in INPUT_TYPES).

const NODE_NAME = "Eden_Save_Param_Dict";
const VALUE_RE = /^value_(\d+)$/;
const KEY_TOOLTIP = "Key for this row. Empty: a dict merges into the top level, any other value uses the upstream output's name. Set on a dict: nests it under this key.";

app.registerExtension({
    name: "Eden.SaveParamDict",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const proto = nodeType.prototype;

        proto.valueInputs = function () {
            return (this.inputs ?? []).filter((i) => VALUE_RE.test(i.name));
        };

        proto.addKeyWidget = function (n) {
            const widget = this.addWidget("text", `key_${n}`, "", () => {});
            widget.tooltip = KEY_TOOLTIP;
            return widget;
        };

        proto.addParamRow = function () {
            const n = Math.max(0, ...this.valueInputs().map((i) => Number(i.name.match(VALUE_RE)[1]))) + 1;
            this.addInput(`value_${n}`, "*");
            this.addKeyWidget(n);
        };

        proto.removeParamRow = function (input) {
            const n = input.name.match(VALUE_RE)[1];
            this.removeInput(this.inputs.indexOf(input));
            const widget = this.widgets?.find((w) => w.name === `key_${n}`);
            if (widget) this.removeWidget(widget);
        };

        proto.syncParamRows = function () {
            let rows = this.valueInputs();
            while (rows.length >= 2 && rows.at(-1).link == null && rows.at(-2).link == null) {
                this.removeParamRow(rows.at(-1));
                rows = this.valueInputs();
            }
            if (!rows.length || rows.at(-1).link != null) this.addParamRow();
            this.setSize([this.size[0], this.computeSize()[1]]);
            this.setDirtyCanvas(true, true);
        };

        const onNodeCreated = proto.onNodeCreated;
        proto.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.syncParamRows();
        };

        const onConnectionsChange = proto.onConnectionsChange;
        proto.onConnectionsChange = function (type, index, connected, linkInfo, slot) {
            const r = onConnectionsChange?.apply(this, arguments);
            if (!this._configuring && type === LiteGraph.INPUT && VALUE_RE.test(slot?.name ?? "")) {
                // Let the link settle before counting connected rows.
                setTimeout(() => this.syncParamRows(), 0);
            }
            return r;
        };

        // Sockets come back from the saved inputs; the key widgets must exist before the default
        // configure assigns widgets_values (save_path first, then key_N in row order).
        const configure = proto.configure;
        proto.configure = function (info) {
            this._configuring = true;
            for (const w of [...(this.widgets ?? [])]) {
                if (w.name?.startsWith("key_")) this.removeWidget(w);
            }
            for (const input of info.inputs ?? []) {
                const m = input.name?.match(VALUE_RE);
                if (m) this.addKeyWidget(m[1]);
            }
            try {
                return configure.apply(this, arguments);
            } finally {
                this._configuring = false;
                // Old workflows also saved a json_string output, since removed.
                while (this.outputs.length > 1) this.removeOutput(this.outputs.length - 1);
                this.syncParamRows();
            }
        };
    },
});
