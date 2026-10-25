import React, { useEffect, useRef } from "react";
import { Animated, StyleSheet, View, useColorScheme } from "react-native";

export function TypingIndicator() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === "dark";

  const dot1 = useRef(new Animated.Value(0.3)).current;
  const dot2 = useRef(new Animated.Value(0.3)).current;
  const dot3 = useRef(new Animated.Value(0.3)).current;

  useEffect(() => {
    const pulse = (dot: Animated.Value, delay: number) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay(delay),
          Animated.timing(dot, { toValue: 1, duration: 400, useNativeDriver: true }),
          Animated.timing(dot, { toValue: 0.3, duration: 400, useNativeDriver: true }),
          Animated.delay(800 - delay),
        ])
      );

    const a1 = pulse(dot1, 0);
    const a2 = pulse(dot2, 200);
    const a3 = pulse(dot3, 400);
    a1.start(); a2.start(); a3.start();
    return () => { a1.stop(); a2.stop(); a3.stop(); };
  }, []);

  const bubbleBg = isDark ? "rgba(10,16,30,0.50)" : "rgba(255,255,255,0.40)";
  const bubbleBorder = isDark ? "rgba(60,100,160,0.20)" : "rgba(200,220,255,0.50)";
  const dotColor = isDark ? "#00d4ff" : "#4a7eb5";

  return (
    <View style={styles.row}>
      <View style={[styles.bubble, { backgroundColor: bubbleBg, borderColor: bubbleBorder }]}>
        {[dot1, dot2, dot3].map((dot, i) => (
          <Animated.View
            key={i}
            style={[styles.dot, { backgroundColor: dotColor, opacity: dot }]}
          />
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    alignItems: "flex-start",
    marginBottom: 10,
    paddingHorizontal: 14,
  },
  bubble: {
    flexDirection: "row",
    gap: 5,
    alignItems: "center",
    paddingHorizontal: 18,
    paddingVertical: 15,
    borderRadius: 20,
    borderTopLeftRadius: 4,
    borderWidth: 1,
    shadowColor: "rgba(0,180,255,0.10)",
    shadowOpacity: 0.15,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 3,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
  },
});
