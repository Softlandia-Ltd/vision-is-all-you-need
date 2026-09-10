import { style } from "@vanilla-extract/css";
import { vars } from "../theme";
import { rem } from "@mantine/core";

export const panel = style({
  display: "flex",
  flexDirection: "column",
  height: "100%",
  minHeight: 0,
});

export const toolbar = style({
  flexShrink: 0,
  padding: `${rem(8)} ${vars.spacing.md}`,
  borderBottom: `1px solid ${vars.colors.gray[3]}`,
  selectors: {
    [vars.darkSelector]: {
      borderBottomColor: vars.colors.dark[4],
    },
  },
});

export const tokens = style({
  flexShrink: 0,
  padding: `${rem(8)} ${vars.spacing.md}`,
  borderBottom: `1px solid ${vars.colors.gray[3]}`,
  selectors: {
    [vars.darkSelector]: {
      borderBottomColor: vars.colors.dark[4],
    },
  },
});

export const scroll = style({
  flex: 1,
  minHeight: 0,
});

export const swatch = style({
  width: rem(18),
  height: rem(18),
  borderRadius: vars.radius.sm,
  border: `2px solid transparent`,
  outlineOffset: rem(2),
  transition: "transform 120ms ease",
  selectors: {
    "&:hover": {
      transform: "scale(1.1)",
    },
    "&[data-active]": {
      outline: `2px solid ${vars.colors.gray[6]}`,
    },
    [`${vars.darkSelector}[data-active]`]: {
      outline: `2px solid ${vars.colors.dark[0]}`,
    },
  },
});

export const pageCard = style({
  flexShrink: 0,
  minWidth: rem(200),
  transition: "width 120ms ease",
});

export const zoomValue = style({
  minWidth: rem(48),
  padding: `${rem(2)} ${rem(4)}`,
  textAlign: "center",
  fontSize: vars.fontSizes.sm,
  color: vars.colors.gray[7],
  borderRadius: vars.radius.sm,
  fontVariantNumeric: "tabular-nums",
  selectors: {
    "&:hover": {
      backgroundColor: vars.colors.gray[1],
    },
    [vars.darkSelector]: {
      color: vars.colors.dark[1],
    },
    [`${vars.darkSelector}:hover`]: {
      backgroundColor: vars.colors.dark[5],
    },
  },
});
