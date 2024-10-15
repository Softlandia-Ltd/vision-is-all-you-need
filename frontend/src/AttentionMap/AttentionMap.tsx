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

  const drawHeatmap = (
    map: number[][] | null,
    ctx: CanvasRenderingContext2D
  ) => {
    if (!map || !ctx) return;

    const canvasWidth = width;
    const canvasHeight = height;
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    const mapHeight = map.length;
    const mapWidth = map[0].length;
    const cellWidth = canvasWidth / mapWidth;
    const cellHeight = canvasHeight / mapHeight;

    for (let row = 0; row < mapHeight; row++) {
      for (let col = 0; col < mapWidth; col++) {
        const attentionValue = map[row][col];

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
  };

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (ctx && tokenMap) {
      drawHeatmap(tokenMap, ctx);
    }
  }, [tokenMap, color, opacity]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        pointerEvents: "auto",
      }}
    />
  );
}
