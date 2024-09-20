import { style } from "@vanilla-extract/css";
import { vars } from "../theme";

export const wrapper = style({
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: "16px",
  width: "100%",
  maxWidth: "100%",
});

export const dropzone = style({
  width: "100%",
  padding: "20px",
  border: `2px dashed ${vars.colors.gray[4]}`,
  borderRadius: vars.radius.md,
  transition: "background-color 0.3s ease",
  backgroundColor: vars.colors.gray[1],

  selectors: {
    "&:hover": {
      backgroundColor: vars.colors.gray[2],
    },
    [vars.darkSelector]: {
      border: `2px dashed ${vars.colors.dark[4]}`,
      backgroundColor: vars.colors.dark[7],
    },
    [vars.darkSelector + " &:hover"]: {
      backgroundColor: vars.colors.dark[6],
    },
  },
});

export const control = style({
  width: "200px",
  display: "block",
  margin: "0 auto",
  textAlign: "center",
  padding: "12px 24px",
  borderRadius: vars.radius.xl,
  backgroundColor: vars.colors.blue[6],
  color: vars.colors.white,
  fontSize: "16px",
  fontWeight: 700,

  selectors: {
    "&:hover": {
      backgroundColor: vars.colors.blue[7],
    },
    [vars.darkSelector]: {
      backgroundColor: vars.colors.blue[8],
    },
    [vars.darkSelector + " &:hover"]: {
      backgroundColor: vars.colors.blue[9],
    },
  },
});
