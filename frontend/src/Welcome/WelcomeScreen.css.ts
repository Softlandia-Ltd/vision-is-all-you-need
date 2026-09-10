import { style } from "@vanilla-extract/css";
import { vars } from "../theme";
import { rem } from "@mantine/core";

export const screen = style({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  height: "100%",
  minHeight: 0,
  overflowY: "auto",
  padding: vars.spacing.md,
});

export const content = style({
  width: "100%",
  maxWidth: rem(720),
});

export const heading = style({
  fontSize: rem(52),
  fontWeight: 700,
  lineHeight: 1.05,
  letterSpacing: rem(-1.4),
  "@media": {
    [vars.smallerThan("sm")]: {
      fontSize: rem(34),
    },
  },
});

export const tagline = style({
  fontSize: rem(24),
  fontWeight: 500,
  color: vars.colors.gray[7],
  selectors: {
    [vars.darkSelector]: {
      color: vars.colors.dark[1],
    },
  },
  "@media": {
    [vars.smallerThan("sm")]: {
      fontSize: rem(18),
    },
  },
});

export const status = style({
  minHeight: rem(96),
});
