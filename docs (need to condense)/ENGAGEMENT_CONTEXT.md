# Engagement Context Integration Guide

## Overview

The Business Meeting Copilot AI coach is designed to provide real-time insights based on the engagement state of meeting participants. This document explains how to integrate engagement state information into conversations with the AI.

## Engagement State Format

When sending messages to the AI, include engagement state information in your conversation context. The AI will use this information to provide tailored, actionable insights.

### Basic Format

Include engagement state as part of the conversation context, either:
1. As a separate message before the user's question
2. Embedded in the user's message
3. As part of the system context

### Example Formats

#### Format 1: Separate Context Message
```
User: "Meeting partner engagement: HIGH - showing active interest and asking questions"
User: "How should I present my proposal now?"
```

#### Format 2: Embedded in Message
```
User: "Meeting partner engagement: LOW - appears distracted, minimal responses. How can I re-engage them?"
```

#### Format 3: Structured Context
```
User: "Engagement State: MEDIUM | Emotional State: Neutral | Participation: Listening but not actively contributing | Current Topic: Budget discussion"
User: "What should I do next?"
```

## Engagement Levels

The AI recognizes and responds to these engagement levels:

### HIGH ENGAGEMENT
- **Indicators**: Active participation, asking questions, showing interest, positive body language, contributing ideas
- **AI Response**: Will suggest capitalizing on momentum, presenting key proposals, seeking commitments, advancing to next steps

### MEDIUM ENGAGEMENT
- **Indicators**: Listening attentively, occasional responses, neutral body language, following along
- **AI Response**: Will suggest building interest, providing compelling details, addressing concerns, creating buy-in

### LOW ENGAGEMENT
- **Indicators**: Distracted, minimal responses, checking phone/email, disengaged body language, not contributing
- **AI Response**: Will suggest re-engagement strategies, changing approach, simplifying message, checking for understanding

### DISAGREEMENT/CONCERN
- **Indicators**: Expressing doubts, asking challenging questions, negative body language, resistance
- **AI Response**: Will suggest acknowledging perspectives, finding common ground, addressing root causes, proposing solutions

### CONFUSION
- **Indicators**: Asking clarifying questions, puzzled expressions, requesting repetition, not following along
- **AI Response**: Will suggest clarification, simplification, providing examples, checking understanding

## Integration Examples

### Example 1: High Engagement Scenario
```
Context: "Meeting partner engagement: HIGH - actively asking questions about the proposal"
User: "They seem very interested. What's my best move right now?"

AI Response: "Great momentum! With high engagement, this is the perfect time to:
1. Present your key proposal details - they're ready to hear specifics
2. Ask for their input on implementation - leverage their interest
3. Suggest next steps or timeline - strike while engagement is high
Consider: 'Based on your questions, I can see you're thinking about implementation. Would you like to discuss the timeline and next steps?'"
```

### Example 2: Low Engagement Scenario
```
Context: "Meeting partner engagement: LOW - appears distracted, checking phone"
User: "They seem disengaged. How do I get their attention back?"

AI Response: "Low engagement detected. Try these re-engagement strategies:
1. Ask a direct question that requires their input - 'What's your take on this approach?'
2. Reference something specific they mentioned earlier - shows you value their input
3. Change the format - suggest a quick break or switch to a different topic
4. Check for understanding - 'I want to make sure this aligns with your priorities'
Consider pausing and asking: 'I notice we've covered a lot. What aspect would be most valuable to focus on right now?'"
```

### Example 3: Disagreement Scenario
```
Context: "Meeting partner engagement: CONCERN - expressing doubts about the timeline"
User: "They're pushing back on the timeline. How should I handle this?"

AI Response: "Disagreement detected. Address concerns while maintaining relationship:
1. Acknowledge their concern directly - 'I understand the timeline feels tight'
2. Explore the root cause - 'What's driving the concern about the timeline?'
3. Find common ground - 'We both want this to succeed. What would make this work?'
4. Propose alternatives - 'Would a phased approach address your concerns?'
Try: 'I hear your concern. Let's explore what would make this timeline workable for you.'"
```

## Implementation in Frontend

### Option 1: Add Engagement State to User Messages
```javascript
// When sending a message, include engagement context
const engagementState = getCurrentEngagementState(); // Your function to detect engagement
const messageWithContext = `Meeting partner engagement: ${engagementState.level} - ${engagementState.description}\n\n${userMessage}`;
```

### Option 2: Include Engagement in System Context
```javascript
// Add engagement state as a separate system message
const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: `Current engagement: ${engagementState}` },
    { role: 'user', content: userMessage }
];
```

### Option 3: Structured Engagement Data
```javascript
// Send structured engagement data
const engagementContext = {
    level: 'HIGH',
    indicators: ['asking questions', 'active participation'],
    emotionalState: 'positive',
    participationLevel: 'high'
};
const contextMessage = `Engagement State: ${JSON.stringify(engagementContext)}`;
```

## Engagement Detection Tips

To provide accurate engagement state, consider monitoring:

1. **Verbal Cues**
   - Frequency of questions
   - Length and detail of responses
   - Tone and enthusiasm
   - Agreement/disagreement indicators

2. **Participation Patterns**
   - Active contribution vs. passive listening
   - Response time and engagement
   - Topic interest indicators

3. **Context Clues**
   - Current meeting topic
   - Stage of conversation
   - Relationship dynamics
   - Previous engagement history

## Best Practices

1. **Update Frequently**: Engagement can change quickly - update context as the meeting progresses
2. **Be Specific**: Provide concrete indicators rather than just "high" or "low"
3. **Include Context**: Mention what's happening in the meeting that relates to engagement
4. **Combine Signals**: Use multiple indicators to assess engagement accurately
5. **Timing Matters**: Provide engagement context right before asking for insights

## Testing Engagement Integration

To test engagement-aware responses:

1. Send a message with engagement context
2. Observe how the AI tailors its response
3. Try different engagement levels with the same question
4. Verify that recommendations change based on engagement state

Example test:
```
Test 1: "Engagement: HIGH - How should I present my proposal?"
Test 2: "Engagement: LOW - How should I present my proposal?"
```

You should see different recommendations for each scenario.
