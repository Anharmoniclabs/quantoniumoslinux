import React from "react";
import { StyleSheet, Text, View, useColorScheme } from "react-native";
import { Message } from "@/contexts/ChatContext";
import { useColors } from "@/hooks/useColors";

interface Props {
  message: Message;
}

export function ChatBubble({ message }: Props) {
  const colors = useColors();
  const colorScheme = useColorScheme();
  const isDark = colorScheme === "dark";
  const isUser = message.role === "user";

  const bubbleBg = isDark
    ? isUser ? "rgba(20,50,90,0.50)" : "rgba(10,16,30,0.50)"
    : isUser ? "rgba(255,255,255,0.50)" : "rgba(255,255,255,0.40)";

  const bubbleBorder = isDark
    ? isUser ? "rgba(0,212,255,0.20)" : "rgba(80,120,180,0.18)"
    : isUser ? "rgba(255,255,255,0.60)" : "rgba(200,220,255,0.50)";

  const textColor = isDark
    ? isUser ? "#d4e8ff" : "#c8d8f0"
    : "#1a2050";

  const timeColor = isDark ? "rgba(150,180,220,0.5)" : "rgba(60,90,160,0.4)";

  const shadowColor = isDark
    ? isUser ? colors.glow : colors.glassShadow
    : colors.glassShadow;

  const time = new Date(message.timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <View style={[styles.row, isUser ? styles.rowRight : styles.rowLeft]}>
      <View
        style={[
          styles.bubble,
          isUser ? styles.bubbleRight : styles.bubbleLeft,
          {
            backgroundColor: bubbleBg,
            borderColor: bubbleBorder,
            shadowColor,
          },
        ]}
      >
        <Text style={[styles.text, { color: textColor }]}>{message.content}</Text>
        <Text style={[styles.time, { color: timeColor }]}>
          {time}
          {message.confidence !== undefined && message.confidence > 0
            ? `  ·  ${(message.confidence * 100).toFixed(0)}%`
            : ""}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    marginBottom: 10,
    paddingHorizontal: 14,
  },
  rowLeft: {
    alignItems: "flex-start",
  },
  rowRight: {
    alignItems: "flex-end",
  },
  bubble: {
    maxWidth: "78%",
    borderRadius: 20,
    borderWidth: 1,
    paddingHorizontal: 16,
    paddingTop: 11,
    paddingBottom: 9,
    shadowOpacity: 0.15,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 3,
  },
  bubbleLeft: {
    borderTopLeftRadius: 4,
  },
  bubbleRight: {
    borderTopRightRadius: 4,
  },
  text: {
    fontSize: 15,
    lineHeight: 22,
    fontFamily: "Inter_400Regular",
  },
  time: {
    fontSize: 10,
    fontFamily: "Inter_400Regular",
    marginTop: 4,
    alignSelf: "flex-end",
  },
});
