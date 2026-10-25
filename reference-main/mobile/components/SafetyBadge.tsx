import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { useColors } from "@/hooks/useColors";

interface Props {
  ok: boolean;
  safeMode: boolean;
}

export function SafetyBadge({ ok, safeMode }: Props) {
  const colors = useColors();
  const color = safeMode ? colors.accent : ok ? colors.safetyGreen : colors.safetyRed;

  return (
    <View style={[styles.badge, { backgroundColor: color + "18" }]}>
      <View style={[styles.dot, { backgroundColor: color }]} />
      <Text style={[styles.label, { color }]}>
        {safeMode ? "Safe" : ok ? "Live" : "Offline"}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  label: {
    fontSize: 12,
    fontWeight: "600",
    fontFamily: "Inter_600SemiBold",
    letterSpacing: 0.2,
  },
});
