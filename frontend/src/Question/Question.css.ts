import { style } from "@vanilla-extract/css";
import { vars } from "../theme";
import { rem } from "@mantine/core";

export const shell = style({
  display: "flex",
  flexDirection: "column",
  gap: rem(6),
  width: "100%",
  padding: rem(10),
  cursor: "text",
  backgroundColor: vars.colors.white,
  border: `1px solid ${vars.colors.gray[3]}`,
  borderRadius: vars.radius.lg,
  transition: "border-color 150ms ease, box-shadow 150ms ease",
  selectors: {
    "&:focus-within": {
      borderColor: vars.colors.coral[6],
      boxShadow: `0 0 0 ${rem(3)} ${vars.colors.coral[1]}`,
    },
    '&[data-disabled]': {
      cursor: "pointer",
    },
    [vars.darkSelector]: {
      backgroundColor: vars.colors.dark[6],
      borderColor: vars.colors.dark[4],
    },
    [`${vars.darkSelector}:focus-within`]: {
      borderColor: vars.colors.coral[5],
      boxShadow: "none",
    },
  },
});

export const shellLg = style({
  padding: rem(14),
  boxShadow: vars.shadows.sm,
});

export const input = style({
  fontSize: vars.fontSizes.md,
  lineHeight: 1.5,
  padding: `0 ${rem(4)}`,
  selectors: {
    "&:disabled": {
      backgroundColor: "transparent",
      color: vars.colors.gray[6],
      opacity: 1,
      cursor: "pointer",
      // Let clicks reach the shell so the whole composer stays clickable
      // while it is locked (the welcome screen opens the file picker).
      pointerEvents: "none",
    },
  },
});

export const inputLg = style({
  fontSize: vars.fontSizes.xl,
});
