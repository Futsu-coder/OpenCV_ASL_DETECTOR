importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

let session = null;
let classes = [];

function iou(box1, box2) {
    const xA = Math.max(box1.x, box2.x);
    const yA = Math.max(box1.y, box2.y);
    const xB = Math.min(box1.x + box1.w, box2.x + box2.w);
    const yB = Math.min(box1.y + box1.h, box2.y + box2.h);
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    return interArea / (box1.w * box1.h + box2.w * box2.h - interArea);
}

self.onmessage = async (e) => {
    const { type, payload } = e.data;

    if (type === 'INIT') {
        classes = payload.classes;
        try {
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
            // เปิดใช้ CPU สูงสุด 4 Core เพื่อช่วยให้ 640 ประมวลผลได้ไวที่สุดเท่าที่จะทำได้
            ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
            
            session = await ort.InferenceSession.create(payload.modelPath, {
                executionProviders: ['wasm']
            });
            self.postMessage({ type: 'READY' });
        } catch (error) {
            self.postMessage({ type: 'ERROR', error: error.message });
        }
    }

    if (type === 'DETECT') {
        if (!session) return;

        try {
            const imgData = payload.imageData;
            const imgSize = 640 * 640; // ⚠️ กลับมาใช้ภาพขนาด 640
            const floatData = new Float32Array(3 * imgSize);
            
            for (let i = 0; i < imgSize; i++) {
                floatData[i] = imgData[i * 4] / 255.0;
                floatData[i + imgSize] = imgData[i * 4 + 1] / 255.0;
                floatData[i + imgSize * 2] = imgData[i * 4 + 2] / 255.0;
            }

            const tensor = new ort.Tensor("float32", floatData, [1, 3, 640, 640]); // ⚠️ สเกล 640
            const results = await session.run({ images: tensor });
            const output = results[session.outputNames[0]].data;

            const numClasses = classes.length;
            const numAnchors = 8400; // ⚠️ จำนวนจุดประมวลผลสำหรับโมเดล 640
            let boxes = [];

            for (let i = 0; i < numAnchors; i++) {
                let maxProb = 0;
                let classId = -1;
                for (let c = 0; c < numClasses; c++) {
                    const prob = output[(4 + c) * numAnchors + i];
                    if (prob > maxProb) { maxProb = prob; classId = c; }
                }

                if (maxProb > 0.60) { // ⚠️ ตั้งเกณฑ์ 60% เพื่อคัดกรองพัดลมหรือผ้าม่านออก
                    const xc = output[0 * numAnchors + i];
                    const yc = output[1 * numAnchors + i];
                    const w = output[2 * numAnchors + i];
                    const h = output[3 * numAnchors + i];
                    boxes.push({ x: xc - w/2, y: yc - h/2, w, h, prob: maxProb, classId });
                }
            }

            boxes.sort((a, b) => b.prob - a.prob);
            const finalBoxes = [];
            while (boxes.length > 0) {
                const best = boxes[0];
                finalBoxes.push(best);
                boxes = boxes.filter(box => iou(best, box) < 0.45);
            }

            self.postMessage({ type: 'RESULT', boxes: finalBoxes });

        } catch (error) {
            console.error("Worker Error details:", error);
            // ⚠️ ป้องกัน AI สลบ ถ้าพังให้ส่งกล่องว่างกลับไป
            self.postMessage({ type: 'RESULT', boxes: [] }); 
        }
    }
};