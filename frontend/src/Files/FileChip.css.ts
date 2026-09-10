import { style } from "@vanilla-extract/css";
import { vars } from "../theme";
import { rem } from "@mantine/core";

export const chip = style({
  padding: `${rem(4)} ${rem(8)}`,
  backgroundColor: vars.colors.white,
  border: `1px solid ${vars.colors.gray[3]}`,
  borderRadius: vars.radius.md,
  selectors: {
    [vars.darkSelector]: {
      backgroundColor: vars.colors.dark[6],
      borderColor: vars.colors.dark[4],
    },
  },
});
