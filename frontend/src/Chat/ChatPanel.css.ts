import { keyframes, style } from "@vanilla-extract/css";
import { vars } from "../theme";
import { rem } from "@mantine/core";

export const panel = style({
  display: "flex",
  flexDirection: "column",
  height: "100%",
  minHeight: 0,
});

export const transcript = style({
  flex: 1,
  minHeight: 0,
});

export const bubble = style({
  alignSelf: "flex-end",
  maxWidth: "85%",
  padding: `${rem(8)} ${rem(12)}`,
  backgroundColor: vars.colors.gray[1],
  borderRadius: vars.radius.lg,
  selectors: {
    [vars.darkSelector]: {
      backgroundColor: vars.colors.dark[5],
    },
  },
});

export const answer = style({
  fontSize: vars.fontSizes.md,
  lineHeight: 1.7,
  overflowWrap: "anywhere",
});

export const composer = style({
  flexShrink: 0,
  padding: vars.spacing.md,
  borderTop: `1px solid ${vars.colors.gray[3]}`,
  selectors: {
    [vars.darkSelector]: {
      borderTopColor: vars.colors.dark[4],
    },
  },
});

const blink = keyframes({
  "0%, 80%, 100%": { opacity: 0.2 },
  "40%": { opacity: 1 },
});

export const typing = style({
  display: "flex",
  gap: rem(4),
  alignItems: "center",
  height: rem(20),
});

export const dot = style({
  width: rem(6),
  height: rem(6),
  borderRadius: "50%",
  backgroundColor: vars.colors.coral[6],
  animation: `${blink} 1.2s infinite ease-in-out`,
  selectors: {
    "&:nth-child(2)": { animationDelay: "0.2s" },
    "&:nth-child(3)": { animationDelay: "0.4s" },
  },
});
