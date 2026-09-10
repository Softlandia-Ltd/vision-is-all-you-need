import { style } from "@vanilla-extract/css";

export const shell = style({
  display: "flex",
  flexDirection: "column",
  height: "100%",
  minHeight: 0,
});

export const main = style({
  flex: 1,
  minHeight: 0,
});
