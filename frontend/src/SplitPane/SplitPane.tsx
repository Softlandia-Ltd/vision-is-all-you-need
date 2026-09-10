import { ReactNode, useEffect, useRef, useState } from "react";
import { useMediaQuery } from "@mantine/hooks";
import * as classes from "./SplitPane.css";

export interface SplitPaneProps {
  left: ReactNode;
  right: ReactNode;
  /** Share of the width given to the left pane on wide screens. */
  defaultRatio?: number;
  min?: number;
  max?: number;
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

export function SplitPane({
  left,
  right,
  defaultRatio = 0.3,
  min = 0.2,
  max = 0.7,
}: SplitPaneProps) {
  const isWide = useMediaQuery("(min-width: 62em)", true);
  const containerRef = useRef<HTMLDivElement>(null);
  const [ratio, setRatio] = useState(defaultRatio);
  const [dragging, setDragging] = useState(false);

  // Keep the cursor consistent and stop text selecting mid-drag.
  useEffect(() => {
    if (!dragging) return;
    const { body } = document;
    const previousCursor = body.style.cursor;
    const previousSelect = body.style.userSelect;
    body.style.cursor = "col-resize";
    body.style.userSelect = "none";
    return () => {
      body.style.cursor = previousCursor;
      body.style.userSelect = previousSelect;
    };
  }, [dragging]);

  // Stacked 50/50 on narrow screens, with no handle to drag.
  if (!isWide) {
    return (
      <div className={classes.stacked}>
        <div className={classes.stackedPane}>{left}</div>
        <div className={classes.stackedDivider} />
        <div className={classes.stackedPane}>{right}</div>
      </div>
    );
  }

  const updateFromPointer = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    setRatio(clamp((clientX - rect.left) / rect.width, min, max));
  };

  return (
    <div
      ref={containerRef}
      className={classes.container}
      style={{ gridTemplateColumns: `${ratio}fr 6px ${1 - ratio}fr` }}
    >
      <div className={classes.pane}>{left}</div>
      <div
        className={classes.handle}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize panels"
        aria-valuenow={Math.round(ratio * 100)}
        aria-valuemin={Math.round(min * 100)}
        aria-valuemax={Math.round(max * 100)}
        tabIndex={0}
        data-dragging={dragging || undefined}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          setDragging(true);
        }}
        onPointerMove={(event) => {
          if (dragging) updateFromPointer(event.clientX);
        }}
        onPointerUp={(event) => {
          event.currentTarget.releasePointerCapture(event.pointerId);
          setDragging(false);
        }}
        onDoubleClick={() => setRatio(defaultRatio)}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") {
            event.preventDefault();
            setRatio((current) => clamp(current - 0.02, min, max));
          } else if (event.key === "ArrowRight") {
            event.preventDefault();
            setRatio((current) => clamp(current + 0.02, min, max));
          }
        }}
      />
      <div className={classes.pane}>{right}</div>
    </div>
  );
}
