import { style } from "@vanilla-extract/css";
import { vars } from "../theme";
import { rem } from "@mantine/core";

export const header = style({
  flexShrink: 0,
  height: rem(56),
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: vars.spacing.md,
  padding: `0 ${vars.spacing.md}`,
  backgroundColor: vars.colors.body,
  borderBottom: `1px solid ${vars.colors.gray[3]}`,
  selectors: {
    [vars.darkSelector]: {
      borderBottomColor: vars.colors.dark[4],
    },
  },
});

export const brand = style({
  minWidth: 0,
});

export const separator = style({
  width: 1,
  height: rem(20),
  backgroundColor: vars.colors.gray[3],
  selectors: {
    [vars.darkSelector]: {
      backgroundColor: vars.colors.dark[4],
    },
  },
  "@media": {
    [vars.smallerThan("xs")]: {
      display: "none",
    },
  },
});

export const title = style({
  fontSize: rem(15),
  fontWeight: 600,
  color: vars.colors.gray[7],
  selectors: {
    [vars.darkSelector]: {
      color: vars.colors.dark[1],
    },
  },
  "@media": {
    [vars.smallerThan("xs")]: {
      display: "none",
    },
  },
});
