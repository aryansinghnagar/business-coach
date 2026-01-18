"""
Context Generator Module

This module generates contextual information from engagement metrics to provide
actionable insights for the AI coaching system. It translates raw engagement
scores and metrics into natural language context that helps the AI suggest
appropriate actions during business meetings.
"""

from dataclasses import dataclass
from typing import List, Optional
from utils.engagement_scorer import EngagementMetrics

# Import EngagementLevel - using lazy import to avoid circular dependency
_EngagementLevel = None

def _get_engagement_level():
    """Lazy import of EngagementLevel to avoid circular dependency."""
    global _EngagementLevel
    if _EngagementLevel is None:
        from engagement_state_detector import EngagementLevel as EL
        _EngagementLevel = EL
    return _EngagementLevel


@dataclass
class EngagementContext:
    """
    Contextual information for AI coaching.
    
    This structure contains natural language descriptions and actionable
    insights derived from engagement metrics.
    """
    summary: str  # Brief summary of engagement state
    level_description: str  # Description of engagement level
    key_indicators: List[str]  # List of key behavioral indicators
    suggested_actions: List[str]  # Suggested actions for the user
    risk_factors: List[str]  # Potential concerns or risk factors
    opportunities: List[str]  # Opportunities to capitalize on


class ContextGenerator:
    """
    Generates contextual information from engagement metrics.
    
    This class translates technical engagement scores into human-readable
    context that can be used by the AI coaching system to provide tailored
    recommendations.
    
    Usage:
        generator = ContextGenerator()
        context = generator.generate_context(score, metrics, level)
    """
    
    def generate_context(
        self,
        score: float,
        metrics: EngagementMetrics,
        level  # EngagementLevel enum (imported lazily to avoid circular dependency)
    ) -> EngagementContext:
        """
        Generate contextual information from engagement data.
        
        Args:
            score: Overall engagement score (0-100)
            metrics: Detailed engagement metrics
            level: Engagement level category
        
        Returns:
            EngagementContext object with generated information
        """
        summary = self._generate_summary(score, level)
        level_description = self._describe_level(level)
        key_indicators = self._identify_key_indicators(metrics, level)
        suggested_actions = self._suggest_actions(score, metrics, level)
        risk_factors = self._identify_risks(score, metrics, level)
        opportunities = self._identify_opportunities(score, metrics, level)
        
        return EngagementContext(
            summary=summary,
            level_description=level_description,
            key_indicators=key_indicators,
            suggested_actions=suggested_actions,
            risk_factors=risk_factors,
            opportunities=opportunities
        )
    
    def _generate_summary(self, score: float, level) -> str:
        """Generate a brief summary of the engagement state."""
        EngagementLevel = _get_engagement_level()
        score_int = int(round(score))
        
        summaries = {
            EngagementLevel.VERY_LOW: f"Very low engagement detected (score: {score_int}/100). The meeting partner appears highly disengaged and may need immediate re-engagement strategies.",
            EngagementLevel.LOW: f"Low engagement detected (score: {score_int}/100). The meeting partner shows signs of distraction or disinterest.",
            EngagementLevel.MEDIUM: f"Moderate engagement detected (score: {score_int}/100). The meeting partner is listening but may need encouragement to participate more actively.",
            EngagementLevel.HIGH: f"High engagement detected (score: {score_int}/100). The meeting partner is actively engaged and receptive.",
            EngagementLevel.VERY_HIGH: f"Very high engagement detected (score: {score_int}/100). The meeting partner is highly engaged and showing strong interest."
        }
        
        return summaries.get(level, f"Engagement level: {score_int}/100")
    
    def _describe_level(self, level) -> str:
        """Provide a detailed description of the engagement level."""
        EngagementLevel = _get_engagement_level()
        descriptions = {
            EngagementLevel.VERY_LOW: "The meeting partner appears highly disengaged, possibly distracted, checking devices, or mentally absent. Immediate intervention may be needed to re-establish connection.",
            EngagementLevel.LOW: "The meeting partner shows low engagement with minimal participation, occasional glances away, or signs of distraction. Consider changing approach or checking in.",
            EngagementLevel.MEDIUM: "The meeting partner is moderately engaged - listening attentively but not actively participating. This is a good time to invite participation or ask questions.",
            EngagementLevel.HIGH: "The meeting partner is actively engaged, showing interest through eye contact, facial expressions, and attention. This is an optimal time to present key points or seek commitments.",
            EngagementLevel.VERY_HIGH: "The meeting partner is highly engaged and showing strong interest. This is an excellent opportunity to advance proposals, seek decisions, or capitalize on momentum."
        }
        
        return descriptions.get(level, "Engagement level is being assessed.")
    
    def _identify_key_indicators(
        self,
        metrics: EngagementMetrics,
        level
    ) -> List[str]:
        EngagementLevel = _get_engagement_level()
        """Identify key behavioral indicators from metrics."""
        indicators = []
        
        # Attention indicators
        if metrics.attention < 40:
            indicators.append("Low attention level - eyes may be closing or unfocused")
        elif metrics.attention > 80:
            indicators.append("High attention level - actively focused")
        
        # Eye contact indicators
        if metrics.eye_contact < 40:
            indicators.append("Limited eye contact - looking away from camera/speaker")
        elif metrics.eye_contact > 75:
            indicators.append("Strong eye contact - maintaining focus")
        
        # Facial expressiveness
        if metrics.facial_expressiveness < 30:
            indicators.append("Minimal facial expressions - appears passive or disengaged")
        elif metrics.facial_expressiveness > 70:
            indicators.append("Active facial expressions - showing interest and engagement")
        
        # Head movement
        if metrics.head_movement < 40:
            indicators.append("Excessive head movement - may indicate distraction or restlessness")
        elif metrics.head_movement > 80:
            indicators.append("Stable head position - focused and attentive")
        
        # Symmetry
        if metrics.symmetry < 50:
            indicators.append("Asymmetric facial features - possible fatigue or distraction")
        
        # Mouth activity
        if metrics.mouth_activity < 30:
            indicators.append("Minimal mouth activity - not speaking or responding")
        elif metrics.mouth_activity > 70:
            indicators.append("Active mouth movement - speaking or showing interest")
        
        return indicators if indicators else ["Standard engagement indicators observed"]
    
    def _suggest_actions(
        self,
        score: float,
        metrics: EngagementMetrics,
        level
    ) -> List[str]:
        EngagementLevel = _get_engagement_level()
        """Generate business-meeting specific suggested actions based on engagement state."""
        actions = []
        
        if level in [EngagementLevel.VERY_LOW, EngagementLevel.LOW]:
            # Low engagement - re-engagement strategies
            actions.append("Pause and ask a direct, open-ended question: 'What's your perspective on this?' or 'How does this align with your priorities?'")
            actions.append("Use their name to regain attention and create personal connection")
            actions.append("Check in professionally: 'I want to make sure this is valuable for you. Should we adjust our focus?'")
            actions.append("Simplify and reframe: Break complex topics into digestible pieces and use concrete examples")
            actions.append("Create urgency or relevance: Connect the topic to their immediate business needs or challenges")
            
            if metrics.eye_contact < 40:
                actions.append("Wait for visual acknowledgment before continuing - make brief eye contact and pause")
            
            if metrics.attention < 40:
                actions.append("Take a strategic pause: 'Let me pause here - what questions do you have so far?'")
            
            if metrics.facial_expressiveness < 30:
                actions.append("Change your energy: Increase vocal variety, use gestures, or introduce a brief story to re-engage")
        
        elif level == EngagementLevel.MEDIUM:
            # Medium engagement - building momentum
            actions.append("Invite active participation: 'I'd love to hear your thoughts on this' or 'What's your experience with similar situations?'")
            actions.append("Provide compelling business context: Share ROI data, success stories, or industry benchmarks")
            actions.append("Address concerns proactively: 'You might be wondering about X - let me address that'")
            actions.append("Build buy-in: Highlight specific benefits, value propositions, or competitive advantages")
            actions.append("Use strategic questions: Ask thought-provoking questions that require engagement to answer")
            actions.append("Create interactive moments: 'Let's think through this together' or 'What would you do in this scenario?'")
        
        elif level in [EngagementLevel.HIGH, EngagementLevel.VERY_HIGH]:
            # High engagement - capitalize on momentum
            actions.append("Present your strongest proposals NOW - this is the optimal moment for key decisions")
            actions.append("Seek commitments while engagement is peak: 'Based on what we've discussed, can we move forward with X?'")
            actions.append("Advance to concrete next steps: Propose specific action items, timelines, and ownership")
            actions.append("Leverage the momentum: Introduce additional value propositions or expand the opportunity")
            actions.append("Request feedback and input: 'What would make this even better?' or 'What else should we consider?'")
            actions.append("Build on their interest: Deepen the discussion in areas where they're most engaged")
            
            if metrics.mouth_activity > 60:
                actions.append("Encourage them to speak: 'I can see you have thoughts on this - please share'")
            
            if metrics.eye_contact > 75:
                actions.append("Make your most important point now - full attention is guaranteed")
        
        # Metric-specific tactical adjustments
        if metrics.head_movement < 50:
            actions.append("Check for environmental distractions or consider if they're multitasking - may need to refocus")
        
        if metrics.symmetry < 50:
            actions.append("Consider offering a brief break - facial asymmetry may indicate fatigue or cognitive load")
        
        if metrics.facial_expressiveness > 70 and level in [EngagementLevel.HIGH, EngagementLevel.VERY_HIGH]:
            actions.append("They're showing strong interest - this is an ideal time to ask for referrals, testimonials, or introductions")
        
        return actions
    
    def _identify_risks(
        self,
        score: float,
        metrics: EngagementMetrics,
        level
    ) -> List[str]:
        EngagementLevel = _get_engagement_level()
        """Identify potential risks or concerns."""
        risks = []
        
        if level in [EngagementLevel.VERY_LOW, EngagementLevel.LOW]:
            risks.append("Risk of losing the meeting partner's attention completely")
            risks.append("Important information may not be received or understood")
            risks.append("Decision-making may be compromised due to low engagement")
            risks.append("Relationship may be negatively impacted if disengagement persists")
        
        if metrics.attention < 30:
            risks.append("Very low attention - partner may be missing critical information")
        
        if metrics.eye_contact < 30:
            risks.append("Minimal eye contact - partner may be multitasking or distracted")
        
        if metrics.head_movement < 40:
            risks.append("Excessive movement may indicate stress, impatience, or distraction")
        
        if level == EngagementLevel.MEDIUM and score < 50:
            risks.append("Engagement is declining - may drop to low if not addressed")
        
        return risks
    
    def _identify_opportunities(
        self,
        score: float,
        metrics: EngagementMetrics,
        level
    ) -> List[str]:
        EngagementLevel = _get_engagement_level()
        """Identify opportunities to capitalize on."""
        opportunities = []
        
        if level in [EngagementLevel.HIGH, EngagementLevel.VERY_HIGH]:
            opportunities.append("Excellent time to present your strongest arguments")
            opportunities.append("Ideal moment to seek commitments or close on decisions")
            opportunities.append("Opportunity to build deeper rapport and trust")
            opportunities.append("Good time to introduce new ideas or proposals")
        
        if metrics.eye_contact > 75:
            opportunities.append("Strong eye contact indicates full attention - perfect for important points")
        
        if metrics.facial_expressiveness > 70:
            opportunities.append("Active expressions suggest interest - capitalize on this engagement")
        
        if metrics.mouth_activity > 60:
            opportunities.append("Partner appears ready to speak - invite their input")
        
        if level == EngagementLevel.MEDIUM and score > 55:
            opportunities.append("Engagement is building - continue with current approach")
        
        return opportunities if opportunities else ["Continue monitoring engagement levels"]
    
    def format_for_ai(self, context: EngagementContext) -> str:
        """
        Format context as a rich, detailed string for AI consumption.
        
        This method formats the engagement context in a comprehensive way that
        provides the AI coach with all necessary information to provide
        actionable, business-meeting specific recommendations.
        
        Args:
            context: EngagementContext object
        
        Returns:
            Formatted string ready for AI prompt with rich business context
        """
        lines = [
            "=== REAL-TIME MEETING PARTNER ENGAGEMENT ANALYSIS ===",
            "",
            f"📊 ENGAGEMENT STATE: {context.summary}",
            f"📈 DETAILED ASSESSMENT: {context.level_description}",
            "",
            "🔍 KEY BEHAVIORAL INDICATORS:",
            *[f"  • {indicator}" for indicator in context.key_indicators],
            "",
            "💡 RECOMMENDED ACTIONS FOR OPTIMIZING THIS MOMENT:",
            *[f"  → {action}" for action in context.suggested_actions],
        ]
        
        if context.risk_factors:
            lines.extend([
                "",
                "⚠️ POTENTIAL RISKS & CONCERNS:",
                *[f"  ⚠ {risk}" for risk in context.risk_factors]
            ])
        
        if context.opportunities:
            lines.extend([
                "",
                "🎯 STRATEGIC OPPORTUNITIES:",
                *[f"  ✓ {opp}" for opp in context.opportunities]
            ])
        
        lines.extend([
            "",
            "=== END ENGAGEMENT ANALYSIS ===",
            "",
            "Use this real-time engagement data to provide specific, actionable coaching advice.",
            "Focus on business-meeting optimization strategies that leverage the current engagement state.",
            "Be concise, practical, and immediately applicable to the meeting context."
        ])
        
        return "\n".join(lines)
