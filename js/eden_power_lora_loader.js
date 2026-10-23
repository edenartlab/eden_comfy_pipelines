import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Frontend for Eden_PowerLoraLoader, adapted from rgthree's Power Lora Loader.
// Each LoRA is one two-row custom widget named lora_N whose value is sent to the backend as-is:
//   row 1: [on] lora name .................. strength (or last sampled value when 🎲 is on)
//   row 2:      [🎲] random range ........... min  max
// value.out is the index of the LoRA's strength output (after the fixed MODEL/CLIP/settings outputs);
// it stays with the LoRA when rows are reordered and is compacted when a LoRA is removed.

const NODE_NAME = "Eden_PowerLoraLoader";
const FIXED_OUTPUTS = 3;
const MAX_LORAS = 16; // keep in sync with MAX_LORAS in lora_utils/power_lora_loader.py
const STEP = 0.05;
const DRAG_STEP = 0.01;
const MARGIN = 10;
const INNER = 3;
const ROW_GAP = 2;
const ACCENT = "#7c7";

let lorasPromise = null;
function getLoras(force = false) {
    if (!lorasPromise || force) {
        lorasPromise = api.fetchApi("/models/loras").then((r) => r.json()).catch(() => []);
    }
    return lorasPromise;
}

function loraLabel(file) {
    return String(file).split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
}

function rowHeight() {
    return LiteGraph.NODE_WIDGET_HEIGHT;
}

function lowQuality() {
    return (app.canvas.ds?.scale || 1) <= 0.5;
}

function fitString(ctx, str, maxWidth) {
    if (ctx.measureText(str).width <= maxWidth) return str;
    let lo = 0, hi = str.length;
    while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        if (ctx.measureText(str.substring(0, mid) + "…").width <= maxWidth) lo = mid;
        else hi = mid - 1;
    }
    return str.substring(0, lo) + "…";
}

function drawToggle(ctx, x, y, h, value) {
    ctx.save();
    if (!lowQuality()) {
        ctx.beginPath();
        ctx.roundRect(x + 4, y + 4, h * 1.5 - 8, h - 8, [h * 0.5]);
        ctx.globalAlpha *= 0.25;
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.fill();
    }
    ctx.restore();
    ctx.save();
    ctx.fillStyle = value === true ? "#89B" : "#888";
    const cx = value === true ? x + h : value === false ? x + h * 0.5 : x + h * 0.75;
    ctx.beginPath();
    ctx.arc(cx, y + h * 0.5, h * 0.36, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return h * 1.5;
}

// Draws "◀ value ▶" ending at `right`; returns [leftArrowX, valueX, rightArrowX, startX].
const ARROW_W = 9;
const NUMBER_W = 32;
const NUMBER_PART_W = ARROW_W + INNER + NUMBER_W + INNER + ARROW_W;
function drawNumber(ctx, right, y, h, value, color) {
    const x = right - NUMBER_PART_W;
    const midY = y + h / 2;
    ctx.fill(new Path2D(`M ${x} ${midY} l ${ARROW_W} 5 l 0 -10 z`));
    const vx = x + ARROW_W + INNER;
    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    if (color) ctx.fillStyle = color;
    ctx.fillText(fitString(ctx, value.toFixed(2), NUMBER_W), vx + NUMBER_W / 2, midY);
    ctx.restore();
    const rx = vx + NUMBER_W + INNER;
    ctx.fill(new Path2D(`M ${rx} ${midY - 5} l ${ARROW_W} 5 l -${ARROW_W} 5 z`));
    return [x, vx, rx];
}

class LoraRowWidget {
    constructor(name, value) {
        this.name = name;
        this.type = "custom";
        this.options = {};
        this.y = 0;
        this.last_y = 0;
        this.value = { on: true, lora: null, strength: 1, random: true, min: 0.5, max: 1, out: 0, ...value };
        this.sampled = null;
        this.hits = [];
        this.active = null;
        this.dragged = false;
    }

    computeSize(width) {
        return [width, rowHeight() * 2 + ROW_GAP];
    }

    draw(ctx, node, width, y) {
        const h = rowHeight();
        const v = this.value;
        const hits = (this.hits = []);
        const y2 = y + h + ROW_GAP;
        const midY = y + h / 2;
        const midY2 = y2 + h / 2;
        const baseAlpha = ctx.globalAlpha;

        ctx.save();
        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
        ctx.beginPath();
        ctx.roundRect(MARGIN, y, width - MARGIN * 2, h * 2 + ROW_GAP, [lowQuality() ? 0 : 8]);
        ctx.fill();
        if (!lowQuality()) ctx.stroke();

        let x = MARGIN;
        const toggleW = drawToggle(ctx, x, y, h, v.on);
        hits.push({ x, w: toggleW, y: 0, h, click: () => (v.on = !v.on) });
        if (lowQuality()) return ctx.restore();

        if (!v.on) ctx.globalAlpha = baseAlpha * 0.4;
        const rowAlpha = ctx.globalAlpha;
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textBaseline = "middle";
        x += toggleW + INNER;
        const right = width - MARGIN - INNER * 2;

        // Row 1: fixed strength, or the last sampled value when randomizing.
        if (v.random) {
            ctx.textAlign = "right";
            ctx.fillStyle = ACCENT;
            ctx.fillText(this.sampled == null ? "🎲 –" : `🎲 ${this.sampled.toFixed(2)}`, right, midY);
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        } else {
            this.addNumberHits(drawNumber(ctx, right, y, h, v.strength), 0, "strength");
        }
        const labelRight = right - NUMBER_PART_W - INNER * 2;
        ctx.textAlign = "left";
        ctx.fillText(fitString(ctx, v.lora ? loraLabel(v.lora) : "None", labelRight - x), x, midY);
        hits.push({ x, w: labelRight - x, y: 0, h, click: (e) => node.chooseLora(e, (file) => node.setRowLora(this, file)) });

        // Row 2: random range.
        let x2 = MARGIN + INNER * 2 + h * 0.5;
        if (!v.random) ctx.globalAlpha = rowAlpha * 0.6;
        const diceW = drawToggle(ctx, x2, y2, h, v.random);
        hits.push({ x: x2, w: diceW, y: h + ROW_GAP, h, click: () => (v.random = !v.random) });
        x2 += diceW + INNER;
        const labelAlpha = ctx.globalAlpha;
        if (!v.random) ctx.globalAlpha = rowAlpha * 0.4;
        const maxParts = drawNumber(ctx, right, y2, h, v.max);
        this.addNumberHits(maxParts, h + ROW_GAP, "max");
        ctx.textAlign = "right";
        ctx.fillText("max", maxParts[0] - INNER, midY2);
        const minRight = maxParts[0] - INNER * 4 - ctx.measureText("max").width;
        const minParts = drawNumber(ctx, minRight, y2, h, v.min);
        this.addNumberHits(minParts, h + ROW_GAP, "min");
        ctx.fillText("min", minParts[0] - INNER, midY2);
        const minLabelX = minParts[0] - INNER * 3 - ctx.measureText("min").width;
        ctx.globalAlpha = labelAlpha;
        ctx.textAlign = "left";
        ctx.fillText(fitString(ctx, "🎲 random", minLabelX - x2), x2, midY2);
        ctx.restore();
    }

    addNumberHits([lx, vx, rx], y, prop) {
        const h = rowHeight();
        const drag = (dx) => (this.value[prop] = Math.round((this.value[prop] + dx * DRAG_STEP) * 100) / 100);
        const step = (dir) => (this.value[prop] = Math.round((this.value[prop] + dir * STEP) * 100) / 100);
        this.hits.push(
            { x: lx, w: ARROW_W, y, h, click: () => step(-1), drag },
            { x: rx, w: ARROW_W, y, h, click: () => step(1), drag },
            {
                x: vx, w: NUMBER_W, y, h, drag,
                click: (e) => app.canvas.prompt(prop, this.value[prop], (s) => {
                    const n = Number(s);
                    if (Number.isFinite(n)) this.value[prop] = n;
                }, e),
            },
        );
    }

    mouse(event, pos, node) {
        const lx = pos[0];
        const ly = pos[1] - this.last_y;
        if (event.type === "pointerdown") {
            this.dragged = false;
            this.active = this.hits.find((a) => lx >= a.x && lx <= a.x + a.w && ly >= a.y && ly <= a.y + a.h) ?? null;
            return !!this.active;
        }
        if (event.type === "pointermove" && this.active?.drag && event.deltaX) {
            this.dragged = true;
            this.active.drag(event.deltaX);
        } else if (event.type === "pointerup") {
            if (this.active?.click && !this.dragged) this.active.click(event);
            this.active = null;
        }
        node.setDirtyCanvas(true, true);
        return true;
    }
}

class HeaderWidget {
    constructor() {
        this.name = "eden_lora_header";
        this.type = "custom";
        this.options = { serialize: false };
        this.serialize = false;
        this.value = "";
        this.y = 0;
        this.last_y = 0;
    }

    computeSize(width) {
        return [width, rowHeight()];
    }

    draw(ctx, node, width, y) {
        if (!node.loraWidgets().length) return;
        const h = rowHeight();
        ctx.save();
        const w = drawToggle(ctx, MARGIN, y, h, node.allLorasState());
        if (!lowQuality()) {
            ctx.globalAlpha *= 0.55;
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
            ctx.textBaseline = "middle";
            ctx.textAlign = "left";
            ctx.fillText("Toggle All", MARGIN + w + INNER, y + h / 2);
            ctx.textAlign = "right";
            ctx.fillText("Strength", width - MARGIN - INNER * 2, y + h / 2);
        }
        ctx.restore();
    }

    mouse(event, pos, node) {
        if (event.type === "pointerdown" && pos[0] <= MARGIN + rowHeight() * 1.5) {
            const on = node.allLorasState() !== true;
            for (const w of node.loraWidgets()) w.value.on = on;
            node.setDirtyCanvas(true, true);
            return true;
        }
        return false;
    }
}

class AddLoraButtonWidget {
    constructor() {
        this.name = "eden_add_lora";
        this.type = "custom";
        this.options = { serialize: false };
        this.serialize = false;
        this.value = "";
        this.y = 0;
        this.last_y = 0;
        this.pressed = false;
    }

    computeSize(width) {
        return [width, rowHeight()];
    }

    draw(ctx, node, width, y) {
        const h = rowHeight();
        ctx.save();
        ctx.fillStyle = this.pressed ? "#444" : LiteGraph.WIDGET_BGCOLOR;
        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
        ctx.beginPath();
        ctx.roundRect(MARGIN, y, width - MARGIN * 2, h, [4]);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("➕ Add Lora", width / 2, y + h / 2);
        ctx.restore();
    }

    mouse(event, pos, node) {
        if (event.type === "pointerdown") this.pressed = true;
        if (event.type === "pointerup" && this.pressed) {
            this.pressed = false;
            node.chooseLora(event, (file) => node.addLora(file));
        }
        node.setDirtyCanvas(true, true);
        return true;
    }
}

app.registerExtension({
    name: "Eden.PowerLoraLoader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const proto = nodeType.prototype;

        proto.loraWidgets = function () {
            return (this.widgets ?? []).filter((w) => w.name?.startsWith("lora_"));
        };

        proto.allLorasState = function () {
            const rows = this.loraWidgets();
            if (rows.every((w) => w.value.on)) return true;
            if (rows.every((w) => !w.value.on)) return false;
            return null;
        };

        proto.chooseLora = async function (event, onChoose) {
            const loras = await getLoras();
            new LiteGraph.ContextMenu(loras, {
                event,
                title: "Choose a LoRA",
                scale: Math.max(1, app.canvas.ds?.scale ?? 1),
                className: "dark",
                callback: (value) => {
                    if (typeof value === "string") onChoose(value);
                    this.setDirtyCanvas(true, true);
                },
            });
        };

        proto.createLoraWidget = function (value, name) {
            this._loraCounter = (this._loraCounter ?? 0) + 1;
            const widget = this.addCustomWidget(new LoraRowWidget(name ?? `lora_${this._loraCounter}`, value));
            const button = this.widgets.find((w) => w.name === "eden_add_lora");
            if (button) {
                this.widgets.splice(this.widgets.indexOf(widget), 1);
                this.widgets.splice(this.widgets.indexOf(button), 0, widget);
            }
            return widget;
        };

        proto.fitHeight = function () {
            this.setSize([this.size[0], Math.max(this.size[1], this.computeSize()[1])]);
            this.setDirtyCanvas(true, true);
        };

        proto.syncStrengthOutputs = function () {
            for (const w of this.loraWidgets()) {
                const output = this.outputs[FIXED_OUTPUTS + w.value.out];
                if (!output) continue;
                output.name = `strength_${w.value.out + 1}`;
                output.label = w.value.lora ? loraLabel(w.value.lora) : output.name;
            }
        };

        proto.addLora = function (file) {
            const count = this.loraWidgets().length;
            if (count >= MAX_LORAS) {
                app.extensionManager?.toast?.add({ severity: "warn", summary: `Power Lora Loader supports at most ${MAX_LORAS} LoRAs`, life: 3000 });
                return;
            }
            this.createLoraWidget({ lora: file, out: count });
            this.addOutput(`strength_${count + 1}`, "FLOAT");
            this.syncStrengthOutputs();
            this.fitHeight();
        };

        proto.setRowLora = function (widget, file) {
            widget.value.lora = file;
            widget.sampled = null;
            this.syncStrengthOutputs();
        };

        proto.removeLora = function (widget) {
            const out = widget.value.out;
            this.removeWidget(widget);
            this.removeOutput(FIXED_OUTPUTS + out);
            for (const w of this.loraWidgets()) {
                if (w.value.out > out) w.value.out--;
            }
            this.syncStrengthOutputs();
            this.setSize([this.size[0], this.computeSize()[1]]);
            this.setDirtyCanvas(true, true);
        };

        const onNodeCreated = proto.onNodeCreated;
        proto.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            // The backend declares MAX_LORAS strength outputs; only show one per LoRA row.
            while (this.outputs.length > FIXED_OUTPUTS) this.removeOutput(this.outputs.length - 1);
            this.addCustomWidget(new HeaderWidget());
            this.addCustomWidget(new AddLoraButtonWidget());
            getLoras();
            this.setSize([Math.max(this.size[0], 360), this.computeSize()[1]]);
        };

        // LoRA rows are dynamic, so recreate them from the saved values before the default
        // configure restores widget values (and the saved strength outputs) onto them.
        const configure = proto.configure;
        proto.configure = function (info) {
            for (const w of this.loraWidgets()) this.removeWidget(w);
            const named = info.widgets_values_named;
            if (named) {
                for (const [name, v] of Object.entries(named)) {
                    if (name.startsWith("lora_") && v?.lora !== undefined) this.createLoraWidget({ ...v }, name);
                }
            } else {
                for (const v of info.widgets_values ?? []) {
                    if (v && typeof v === "object" && v.lora !== undefined) this.createLoraWidget({ ...v });
                }
            }
            const r = configure.apply(this, arguments);
            this.syncStrengthOutputs();
            return r;
        };

        const onExecuted = proto.onExecuted;
        proto.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const sampled = message?.sampled_strengths;
            if (!sampled) return;
            for (const w of this.loraWidgets()) w.sampled = sampled[w.value.out] ?? null;
            this.setDirtyCanvas(true, true);
        };

        // Right-click a LoRA row for move / remove.
        const getExtraMenuOptions = proto.getExtraMenuOptions;
        proto.getExtraMenuOptions = function (canvas, options) {
            const r = getExtraMenuOptions?.apply(this, arguments);
            const y = canvas.graph_mouse[1] - this.pos[1];
            const rows = this.loraWidgets();
            const widget = rows.find((w) => y >= w.last_y && y <= w.last_y + w.computeSize(this.size[0])[1]);
            if (!widget) return r;
            const i = rows.indexOf(widget);
            const swap = (other) => {
                const a = this.widgets.indexOf(widget);
                const b = this.widgets.indexOf(other);
                [this.widgets[a], this.widgets[b]] = [this.widgets[b], this.widgets[a]];
                this.setDirtyCanvas(true, true);
            };
            const name = widget.value.lora ? loraLabel(widget.value.lora) : "LoRA";
            options.unshift(
                { content: `${widget.value.on ? "⚫ Disable" : "🟢 Enable"} ${name}`, callback: () => (widget.value.on = !widget.value.on) },
                { content: "⬆️ Move Up", disabled: i === 0, callback: () => swap(rows[i - 1]) },
                { content: "⬇️ Move Down", disabled: i === rows.length - 1, callback: () => swap(rows[i + 1]) },
                { content: `🗑️ Remove ${name}`, callback: () => this.removeLora(widget) },
                null,
            );
            return r;
        };

        proto.refreshComboInNode = function () {
            getLoras(true);
        };
    },
});
