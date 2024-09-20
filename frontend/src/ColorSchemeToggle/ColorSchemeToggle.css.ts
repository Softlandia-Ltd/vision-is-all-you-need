import { style } from "@vanilla-extract/css";
import { vars } from "./../theme"; // Assuming your theme setup is here

export const icon = style({
  width: "22px",
  height: "22px",
});

export const dark = style({
  selectors: {
    [vars.darkSelector]: {
      display: "none", // Hidden in dark mode
    },
    [vars.lightSelector]: {
      display: "block", // Visible in light mode
    },
  },
});

export const light = style({
  selectors: {
    [vars.lightSelector]: {
      display: "none", // Hidden in light mode
    },
    [vars.darkSelector]: {
      display: "block", // Visible in dark mode
    },
  },
});
