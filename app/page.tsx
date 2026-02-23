"use client";

import { useEffect, useRef, useState } from "react";

export default function UltimatePerformancePage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const workerRef = useRef<Worker | null>(null);
  const isWorkerBusy = useRef(false);
  const boxesRef = useRef<any[]>([]);
  const smoothBoxRef = useRef<any>(null);
  
  // ใช้ useRef แทน useState ป้องกันลูปนรก Network ระเบิด
  const classesRef = useRef<string[]>([]);

  const [status, setStatus] = useState("กำลังตั้งค่าระบบสมองกล...");
  const [detectText, setDetectText] = useState("-");
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    const worker = new Worker("/yolo-worker.js");
    workerRef.current = worker;

    const initAI = async () => {
      try {
        const resJson = await fetch("/models/classes.json");
        const classNames = await resJson.json();
        classesRef.current = classNames;

        // ⚠️ โหลดโมเดล 640 ตัวออริจินัล
        worker.postMessage({
          type: "INIT",
          payload: { modelPath: "/models/yolo_asl.onnx", classes: classNames } 
        });
      } catch (e) {
        setStatus("โหลดรายชื่อคลาสไม่สำเร็จ!");
      }
    };

    worker.onmessage = (e) => {
      const { type, boxes, error } = e.data;
      if (type === "READY") {
        setStatus("🔥 ระบบประมวลผลสูงสุดพร้อมใช้งาน! (กด OPEN_CAM)");
      } else if (type === "RESULT") {
        boxesRef.current = boxes;
        if (boxes.length > 0) {
          const best = boxes[0];
          setDetectText(`${classesRef.current[best.classId] || ""} (${(best.prob * 100).toFixed(0)}%)`);
        } else {
          setDetectText("-");
        }
        isWorkerBusy.current = false; // รับผลลัพธ์แล้ว ปลดล็อกให้ส่งภาพใหม่ได้
      } else if (type === "ERROR") {
        setStatus(`Error: ${error}`);
      }
    };

    initAI();

    return () => worker.terminate();
  }, []); // ⚠️ ปล่อยว่างไว้เพื่อป้องกัน Infinite Loop

  function masterLoop() {
    if (!videoRef.current || !canvasRef.current || !streamRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;

    if (video.videoWidth > 0) {
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (!isWorkerBusy.current && workerRef.current) {
        isWorkerBusy.current = true;
        const offCanvas = document.createElement("canvas");
        offCanvas.width = 640;  // ⚠️ เปลี่ยนกลับเป็น 640
        offCanvas.height = 640; // ⚠️ เปลี่ยนกลับเป็น 640
        const offCtx = offCanvas.getContext("2d", { willReadFrequently: true })!;
        
        offCtx.drawImage(video, 0, 0, 640, 640); // ⚠️ 640
        
        const imageData = offCtx.getImageData(0, 0, 640, 640).data;
        workerRef.current.postMessage({ type: "DETECT", payload: { imageData } });
      }

      const scaleX = canvas.width / 640; // ⚠️ สเกลตาม 640
      const scaleY = canvas.height / 640; // ⚠️ สเกลตาม 640
      const currentBoxes = boxesRef.current;

      if (currentBoxes.length > 0) {
        const target = currentBoxes[0];
        if (!smoothBoxRef.current || smoothBoxRef.current.classId !== target.classId) {
          smoothBoxRef.current = { ...target };
        } else {
          // เพิ่มความเร็วการสไลด์กล่องนิดนึงให้ตามมือทัน (speed = 0.4)
          const speed = 0.4; 
          smoothBoxRef.current.x += (target.x - smoothBoxRef.current.x) * speed;
          smoothBoxRef.current.y += (target.y - smoothBoxRef.current.y) * speed;
          smoothBoxRef.current.w += (target.w - smoothBoxRef.current.w) * speed;
          smoothBoxRef.current.h += (target.h - smoothBoxRef.current.h) * speed;
          smoothBoxRef.current.prob = target.prob;
        }

        const box = smoothBoxRef.current;
        const rx = box.x * scaleX, ry = box.y * scaleY, rw = box.w * scaleX, rh = box.h * scaleY;
        const label = classesRef.current[box.classId];

        ctx.strokeStyle = "#00FFAA";
        ctx.lineWidth = 5;
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.fillStyle = "#00FFAA";
        ctx.fillRect(rx, ry - 35, rw, 35);
        ctx.fillStyle = "#000000";
        ctx.font = "bold 24px Arial";
        ctx.fillText(`${label} ${(box.prob * 100).toFixed(0)}%`, rx + 8, ry - 8);
      } else {
        smoothBoxRef.current = null;
      }
    }

    if (streamRef.current) {
      requestAnimationFrame(masterLoop);
    }
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      streamRef.current = stream;
      videoRef.current!.srcObject = stream;
      await videoRef.current!.play();

      setIsStreaming(true);
      setStatus("🟢 กำลังตรวจจับความแม่นยำระดับ 640 MAX...");
      requestAnimationFrame(masterLoop);
    } catch (err) {
      alert("เปิดกล้องไม่ได้ครับ");
    }
  }

  function stopCamera() {
    setIsStreaming(false);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    boxesRef.current = [];
    smoothBoxRef.current = null;
    setDetectText("-");
    setStatus("🔴 กล้องปิดแล้ว");
  }

  return (
    <div className="app-root bg-gray-950 min-h-screen p-4 text-white flex flex-col justify-center items-center">
      <div className="text-center mb-6">
        <h1 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-cyan-500 mb-2">
          ⚡ ASL Object Detection MAX (640p)
        </h1>
        <p className="text-gray-400">{status}</p>
      </div>

      <div className="flex flex-col md:flex-row gap-6 items-start">
        <div className="relative border-4 border-gray-800 rounded-2xl overflow-hidden bg-black shadow-2xl shadow-green-900/20" style={{ maxWidth: '640px', width: '100%' }}>
          <canvas ref={canvasRef} className="w-full h-auto" />
          <video ref={videoRef} className="hidden" playsInline muted />
        </div>

        <div className="flex flex-col gap-4 w-full md:w-64">
          <div className="bg-gray-900 border border-gray-800 p-6 rounded-2xl text-center shadow-lg">
            <h3 className="text-gray-400 text-sm font-bold tracking-widest mb-2">RESULT</h3>
            <div className="text-5xl font-black text-white bg-gray-800 py-4 rounded-xl">
              {detectText.split(' ')[0] || "-"}
            </div>
            {detectText !== "-" && (
              <div className="text-green-400 font-bold mt-2">
                CONFIDENCE: {detectText.split(' ')[1]}
              </div>
            )}
          </div>

          {!isStreaming ? (
            <button className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 text-white font-bold py-4 rounded-xl transition-all shadow-lg shadow-green-500/30 text-lg" onClick={startCamera}>
              ▶ START SCANNER
            </button>
          ) : (
            <button className="bg-gradient-to-r from-red-600 to-rose-700 hover:from-red-500 hover:to-rose-600 text-white font-bold py-4 rounded-xl transition-all shadow-lg shadow-red-500/30 text-lg" onClick={stopCamera}>
              ■ STOP CAMERA
            </button>
          )}
        </div>
      </div>
    </div>
  );
}