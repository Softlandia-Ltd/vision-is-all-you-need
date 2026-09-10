import { style } from "@vanilla-extract/css";
import { vars } from "../theme";

export const container = style({
  display: "grid",
  width: "100%",
  height: "100%",
  minHeight: 0,
});

export const pane = style({
  minWidth: 0,
  minHeight: 0,
  height: "100%",
  overflow: "hidden",
});

export const handle = style({
  position: "relative",
  cursor: "col-resize",
  touchAction: "none",
  backgroundColor: "transparent",
  selectors: {
    "&::after": {
      content: '""',
      position: "absolute",
      top: 0,
      bottom: 0,
      left: "50%",
      width: 1,
      transform: "translateX(-50%)",
      backgroundColor: vars.colors.gray[3],
      transition: "background-color 150ms ease, width 150ms ease",
    },
    "&:hover::after": {
      width: 3,
      backgroundColor: vars.colors.coral[6],
    },
    "&:focus-visible::after": {
      width: 3,
      backgroundColor: vars.colors.coral[6],
    },
    "&[data-dragging]::after": {
      width: 3,
      backgroundColor: vars.colors.coral[6],
    },
    "&:focus-visible": {
      outline: "none",
    },
    [`${vars.darkSelector}::after`]: {
      backgroundColor: vars.colors.dark[4],
    },
  },
});

export const stacked = style({
  display: "flex",
  flexDirection: "column",
  width: "100%",
  height: "100%",
  minHeight: 0,
});

export const stackedPane = style({
  flex: "1 1 50%",
  minHeight: 0,
  overflow: "hidden",
});

export const stackedDivider = style({
  flexShrink: 0,
  height: 1,
  backgroundColor: vars.colors.gray[3],
  selectors: {
    [vars.darkSelector]: {
      backgroundColor: vars.colors.dark[4],
    },
  },
});
