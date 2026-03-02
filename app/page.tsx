"use client";

import { useEffect, useRef, useState } from "react";

interface DetectHistory {
  id: number;
  image: string;
  label: string;
  confidence: number;
  timeStr: string;
}

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
  const lastCaptureRef = useRef({ label: "", time: 0 });

  const [status, setStatus] = useState("กำลังเตรียมระบบแปลภาษา...");
  const [detection, setDetection] = useState({ label: "-", confidence: 0, isDetecting: false });
  const [isStreaming, setIsStreaming] = useState(false);
  const [history, setHistory] = useState<DetectHistory[]>([]);
  const [isModelReady, setIsModelReady] = useState(false);
  
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");

  useEffect(() => {
    const worker = new Worker("/yolo-worker.js");
    workerRef.current = worker;

    const initAI = async () => {
      try {
        const resJson = await fetch("/models/classes.json");
        const classNames = await resJson.json();
        classesRef.current = classNames;
        worker.postMessage({
          type: "INIT",
          payload: { modelPath: "/models/yolo_asl.onnx", classes: classNames } 
        });
      } catch (e) {
        setStatus("โหลดข้อมูลไม่สำเร็จ กรุณารีเฟรชหน้าเว็บ");
      }
    };

    worker.onmessage = (e) => {
      const { type, boxes, error } = e.data;
      if (type === "READY") {
        setStatus("กล้องพร้อมแล้วกดปุ่มได้");
        setIsModelReady(true);
      } else if (type === "RESULT") {
        boxesRef.current = boxes;
        lastDetectTimeRef.current = Date.now(); 
        if (boxes.length > 0) {
          const best = boxes[0];
          if (best.prob >= 0.50) {
            setDetection({ 
              label: classesRef.current[best.classId], 
              confidence: Math.round(best.prob * 100), 
              isDetecting: false 
            });
          } else {
            setDetection({ label: "กำลังวิเคราะห์...", confidence: 0, isDetecting: true });
          }
        } else {
          setDetection({ label: "-", confidence: 0, isDetecting: false });
        }
        isWorkerBusy.current = false; 
      } else if (type === "ERROR") {
        setStatus(`ข้อผิดพลาด: ${error}`);
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
        offCanvas.width = 416; 
        offCanvas.height = 416; 
        const offCtx = offCanvas.getContext("2d", { willReadFrequently: true })!;
        offCtx.drawImage(video, 0, 0, 416, 416); 
        const imageData = offCtx.getImageData(0, 0, 416, 416).data; 
        workerRef.current.postMessage({ type: "DETECT", payload: { imageData } });
      }
      if (Date.now() - lastDetectTimeRef.current > 1000) {
        boxesRef.current = [];
        smoothBoxRef.current = null;
        setDetection({ label: "-", confidence: 0, isDetecting: false });
      }

      const scaleX = canvas.width / 416; 
      const scaleY = canvas.height / 416; 
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
        const labelText = classesRef.current[box.classId];
        const isConfident = box.prob >= 0.50;
        const boxColor = isConfident ? "#10B981" : "#F59E0B"; 
        const displayText = isConfident ? `${labelText} ${Math.round(box.prob * 100)}%` : `กำลังวิเคราะห์...`;
        
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 6; 
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.fillStyle = boxColor;
        ctx.fillRect(rx, ry - 40, rw, 40);
        ctx.fillStyle = "#FFFFFF"; 
        ctx.font = "bold 24px Arial";
        ctx.fillText(displayText, rx + 10, ry - 12);
        
        if (box.prob >= 0.60) {
          const now = Date.now();
          const timeSinceLastCapture = now - lastCaptureRef.current.time;
          
          if (labelText !== lastCaptureRef.current.label || timeSinceLastCapture > 3000) {
            lastCaptureRef.current = { label: labelText, time: now };
            const imageBase64 = canvas.toDataURL("image/jpeg", 0.7);
            
            setHistory(prev => {
              const newRecord: DetectHistory = {
                id: now,
                image: imageBase64,
                label: labelText,
                confidence: Math.round(box.prob * 100),
                timeStr: new Date(now).toLocaleTimeString('th-TH')
              };
              return [newRecord, ...prev].slice(0, 12);
            });
          }
        }
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
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode } });
      streamRef.current = stream;
      videoRef.current!.srcObject = stream;
      await videoRef.current!.play();
      setIsStreaming(true);
      requestAnimationFrame(masterLoop);
    } catch (err) {
      alert("ไม่สามารถเปิดกล้องได้ กรุณาอนุญาตการเข้าถึงกล้องถ่ายรูป");
    }
  }

  function stopCamera() {
    setIsStreaming(false);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    boxesRef.current = [];
    smoothBoxRef.current = null;
    setDetection({ label: "-", confidence: 0, isDetecting: false });
    setStatus("ปิดกล้องแล้ว");
  }

  async function toggleCamera() {
    const newMode = facingMode === "user" ? "environment" : "user";
    setFacingMode(newMode);

    if (isStreaming) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: newMode } });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err) {
        alert("ไม่สามารถเปิดกล้องเพื่อสลับได้");
        stopCamera();
      }
    }
  }

  function clearHistory() {
    setHistory([]);
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col p-4 md:p-8 font-sans selection:bg-green-500 selection:text-white">
      <header className="w-full max-w-4xl mx-auto text-center mb-6 md:mb-10 mt-4">
        <h1 className="text-3xl md:text-5xl font-black text-white mb-3 tracking-tight">
          แปลภาษามือ <span className="text-green-400">ASL</span> 🤟
        </h1>
        <p className="text-gray-400 text-sm md:text-base bg-gray-900 inline-block px-4 py-2 rounded-full border border-gray-800">
          {status}
        </p>
      </header>

      <main className="flex flex-col md:flex-row w-full max-w-5xl mx-auto gap-6 items-stretch justify-center">        
        <div className="flex-1 w-full flex flex-col items-center">
          <div className={`relative w-full max-w-2xl aspect-[4/3] rounded-3xl overflow-hidden bg-gray-900 border-4 transition-colors duration-300 shadow-2xl ${isStreaming ? 'border-green-500 shadow-green-900/20' : 'border-gray-800'}`}>
            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 p-6 text-center">
                <svg className="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                <p>หน้าจอกล้องจะแสดงที่นี่</p>
              </div>
            )}
            <canvas ref={canvasRef} className="w-full h-full object-cover" />
            <video ref={videoRef} className="hidden" playsInline muted />
          </div>
        </div>

        <div className="w-full md:w-80 flex flex-col gap-4 shrink-0">
          <div className="bg-gray-900 border border-gray-800 p-6 rounded-3xl text-center shadow-lg flex-1 flex flex-col justify-center min-h-[160px]">
            <h3 className="text-gray-500 text-xs font-bold tracking-[0.2em] mb-4 uppercase">คำแปล / Result</h3>
            <div className={`text-5xl md:text-6xl font-black uppercase transition-colors duration-300 ${
                detection.isDetecting ? "text-yellow-400 text-3xl md:text-4xl" : 
                detection.label !== "-" ? "text-white" : "text-gray-700"
              }`}>
              {detection.label}
            </div>
            {detection.label !== "-" && !detection.isDetecting && (
              <div className="text-green-400 font-bold mt-4 bg-green-950/30 inline-block self-center px-3 py-1 rounded-full text-sm border border-green-900/50">
                ความแม่นยำ: {detection.confidence}%
              </div>
            )}
          </div>

          <div className="mt-2 md:mt-auto flex flex-col gap-3">
            {!isStreaming ? (
              <button 
                className={`w-full font-black py-5 rounded-2xl transition-all shadow-lg text-xl flex items-center justify-center gap-2 ${
                  isModelReady 
                    ? "bg-green-500 hover:bg-green-400 text-gray-950 shadow-green-500/20 active:scale-95 cursor-pointer" 
                    : "bg-gray-800 text-gray-500 cursor-not-allowed shadow-none"
                }`} 
                onClick={startCamera}
                disabled={!isModelReady}
              >
                {isModelReady ? (
                  <>
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    เปิดกล้องเริ่มแปล
                  </>
                ) : (
                  <>
                    <svg className="animate-spin w-6 h-6 text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    กำลังโหลด AI...
                  </>
                )}
              </button>
            ) : (
              <button 
                className="w-full bg-red-500 hover:bg-red-400 text-white font-bold py-5 rounded-2xl transition-all shadow-lg shadow-red-500/20 text-xl active:scale-95 flex items-center justify-center gap-2" 
                onClick={stopCamera}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" /></svg>
                หยุดกล้อง
              </button>
            )}

            <button 
              className="w-full bg-gray-800 hover:bg-gray-700 text-gray-300 font-bold py-3 rounded-2xl transition-all active:scale-95 flex items-center justify-center gap-2"
              onClick={toggleCamera}
              disabled={!isModelReady}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
              สลับกล้อง ({facingMode === "user" ? "หน้า" : "หลัง"})
            </button>
          </div>
        </div>
      </main>

      {history.length > 0 && (
        <div className="w-full max-w-5xl mx-auto mt-10 animate-fade-in">
          <div className="flex justify-between items-center mb-4 px-2">
            <h3 className="text-xl font-bold text-white flex items-center gap-2">
              ประวัติการทำภาษามือล่าสุด
            </h3>
            <button 
              onClick={clearHistory}
              className="text-sm text-gray-400 hover:text-red-400 transition-colors"
            >
              ลบประวัติทั้งหมด
            </button>
          </div>
          
          <div className="flex gap-4 overflow-x-auto pb-4 custom-scrollbar snap-x">
            {history.map((item) => (
              <div 
                key={item.id} 
                className="min-w-[160px] max-w-[160px] bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden shadow-lg flex-shrink-0 snap-start transition-transform hover:scale-105"
              >
                <img 
                  src={item.image} 
                  alt={`Sign ${item.label}`} 
                  className="w-full h-[120px] object-cover bg-black" 
                />
                <div className="p-3 text-center bg-gray-900">
                  <div className="text-2xl font-black text-white">{item.label}</div>
                  <div className="text-xs text-green-400 font-bold mt-1">
                    {item.confidence}% • {item.timeStr}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}      
      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar { height: 8px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #1f2937; border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #4b5563; }
        .animate-fade-in { animation: fadeIn 0.5s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
      `}} />
    </div>
  );
}