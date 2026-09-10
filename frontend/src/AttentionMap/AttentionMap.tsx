import { useLayoutEffect, useRef } from "react";

export interface AttentionMapProps {
  tokenMap: number[][] | null;
  width: number;
  height: number;
  color: "red" | "blue" | "green";
  opacity: number;
}

export function AttentionMap({
  tokenMap,
  width,
  height,
  color,
  opacity,
}: AttentionMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !tokenMap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    const mapHeight = tokenMap.length;
    const mapWidth = tokenMap[0].length;
    const cellWidth = width / mapWidth;
    const cellHeight = height / mapHeight;

    for (let row = 0; row < mapHeight; row++) {
      for (let col = 0; col < mapWidth; col++) {
        const attentionValue = tokenMap[row][col];

        const intensity = Math.min(Math.max(attentionValue * 255, 0), 255);
        const red = color === "red" ? 255 : 0;
        const green = color === "green" ? 255 : 0;
        const blue = color === "blue" ? 255 : 0;
        ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, ${
          (intensity / 255) * (opacity / 100)
        })`;

        ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight);
      }
    }
  }, [tokenMap, color, opacity, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        position: "absolute",
        inset: 0,
        // Internal resolution stays at the page's natural size; CSS stretches
        // the canvas so it tracks the image at any zoom level.
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
    />
  );
}
