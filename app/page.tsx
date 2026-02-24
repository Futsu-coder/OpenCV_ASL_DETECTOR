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
  
  const lastDetectTimeRef = useRef<number>(Date.now());
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

        // ⚠️ โหลดไฟล์ yolo_asl_416.onnx (อย่าลืมเปลี่ยนชื่อไฟล์โมเดลในโฟลเดอร์ให้ตรงด้วยนะครับ!)
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
        setStatus("🔥 ระบบ Local 416 MAX พร้อมใช้งาน! (กด OPEN_CAM)");
      } else if (type === "RESULT") {
        boxesRef.current = boxes;
        lastDetectTimeRef.current = Date.now(); 

        if (boxes.length > 0) {
          const best = boxes[0];
          // 🌟 โชว์ผลลัพธ์ที่ฝั่งขวา เฉพาะตอนที่มั่นใจเกิน 50% เท่านั้น
          if (best.prob >= 0.50) {
            setDetectText(`${classesRef.current[best.classId]} (${(best.prob * 100).toFixed(0)}%)`);
          } else {
            setDetectText(`DETECTING...`); // กำลังเพ่งมืออยู่
          }
        } else {
          setDetectText("-");
        }
        isWorkerBusy.current = false; 
      } else if (type === "ERROR") {
        setStatus(`Error: ${error}`);
      }
    };

    initAI();

    return () => worker.terminate();
  }, []);

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
        offCanvas.width = 416; // ⚠️ แก้เป็น 416
        offCanvas.height = 416; // ⚠️ แก้เป็น 416
        const offCtx = offCanvas.getContext("2d", { willReadFrequently: true })!;
        offCtx.drawImage(video, 0, 0, 416, 416); // ⚠️ แก้เป็น 416
        const imageData = offCtx.getImageData(0, 0, 416, 416).data; // ⚠️ แก้เป็น 416
        workerRef.current.postMessage({ type: "DETECT", payload: { imageData } });
      }

      if (Date.now() - lastDetectTimeRef.current > 1000) {
        boxesRef.current = [];
        smoothBoxRef.current = null;
        setDetectText("-");
      }

      const scaleX = canvas.width / 416; // ⚠️ สเกล 416
      const scaleY = canvas.height / 416; // ⚠️ สเกล 416
      const currentBoxes = boxesRef.current;

      if (currentBoxes.length > 0) {
        const target = currentBoxes[0];
        if (!smoothBoxRef.current || smoothBoxRef.current.classId !== target.classId) {
          smoothBoxRef.current = { ...target };
        } else {
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
        
        // 🌟 ระบบสีของกล่อง (มั่นใจสีเขียว ไม่มั่นใจสีส้ม)
        const isConfident = box.prob >= 0.50;
        const boxColor = isConfident ? "#00FFAA" : "#FFAA00";
        const displayText = isConfident ? `${label} ${(box.prob * 100).toFixed(0)}%` : `Detecting...`;

        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 5;
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.fillStyle = boxColor;
        ctx.fillRect(rx, ry - 35, rw, 35);
        ctx.fillStyle = "#000000";
        ctx.font = "bold 20px Arial";
        ctx.fillText(displayText, rx + 8, ry - 8);
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
      setStatus("🟢 ระบบทำงานเต็มประสิทธิภาพ (416 MAX)...");
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
          ⚡ ASL Object Detection (Local 416)
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
            {/* โชว์แค่ตัวหนังสือเพียวๆ ไม่ต้องมี % ถ้ากำลัง Detecting */}
            <div className={`text-4xl font-black ${detectText === "DETECTING..." ? "text-yellow-400" : "text-white"} bg-gray-800 py-4 rounded-xl`}>
              {detectText.split(' ')[0] || "-"}
            </div>
            {detectText !== "-" && detectText !== "DETECTING..." && (
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