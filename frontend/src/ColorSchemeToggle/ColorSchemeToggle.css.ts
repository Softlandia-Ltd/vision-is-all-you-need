import { style } from "@vanilla-extract/css";
import { vars } from "./../theme";

export const icon = style({
  width: "22px",
  height: "22px",
});

export const dark = style({
  selectors: {
    [vars.darkSelector]: {
      display: "none",
    },
    [vars.lightSelector]: {
      display: "block",
    },
  },
});

export const light = style({
  selectors: {
    [vars.lightSelector]: {
      display: "none",
    },
    [vars.darkSelector]: {
      display: "block",
    },
  },
});
