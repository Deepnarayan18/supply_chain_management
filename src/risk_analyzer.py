"""
Risk Analyzer Module
Combines weather data, GDELT event data, and Gemini AI analysis
to calculate comprehensive risk scores
"""
from typing import Dict, List
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MAJOR_PORTS, RISK_THRESHOLDS
from gdelt_service import GDELTService
from gemini_service import GeminiService


class RiskAnalyzer:
    """Analyzes and scores supply chain risks for ports using AI"""
    
    def __init__(self):
        self.risk_thresholds = RISK_THRESHOLDS
        self.gdelt_service = GDELTService()
        self.gemini_service = GeminiService()
        
    def calculate_comprehensive_risk(
        self, 
        port_name: str, 
        weather_risk: Dict, 
        gdelt_events: List[Dict]
    ) -> Dict:
        """
        Calculate comprehensive risk score combining weather and GDELT events
        Uses Gemini AI for intelligent analysis
        
        Args:
            port_name: Name of the port
            weather_risk: Weather risk analysis
            gdelt_events: List of GDELT events affecting the port
            
        Returns:
            Comprehensive risk analysis with AI-generated insights
        """
        # Get GDELT events for this port
        port_events = [e for e in gdelt_events if e.get("location", "").lower() == port_name.lower()]
        
        # Calculate GDELT risk score
        gdelt_risk = self.gdelt_service.calculate_gdelt_risk_score(port_events)
        
        # Use Gemini AI to analyze combined risks
        try:
            ai_analysis = self.gemini_service.analyze_supply_chain_risk(
                location=port_name,
                gdelt_data=gdelt_risk,
                weather_data=weather_risk
            )
        except Exception as e:
            print(f"Error in AI analysis for {port_name}: {e}")
            # Fallback to basic analysis
            ai_analysis = self._basic_risk_analysis(port_name, weather_risk, gdelt_risk)
        
        # Calculate combined risk score
        weather_score = weather_risk.get("risk_score", 0)
        gdelt_score = gdelt_risk.get("risk_score", 0)
        
        # Weight: 40% weather, 60% GDELT (human factors often more impactful)
        total_risk = (weather_score * 0.4) + (gdelt_score * 0.6)
        total_risk = min(total_risk, 100)
        
        # Map AI risk level to color
        risk_level = ai_analysis.get("risk_level", "Low").lower()
        if risk_level == "critical":
            color = "#d32f2f"
        elif risk_level == "high":
            color = "#f57c00"
        elif risk_level == "medium":
            color = "#fbc02d"
        else:
            color = "#388e3c"
        
        # Compile risk factors
        risk_factors = []
        risk_factors.extend(weather_risk.get("factors", []))
        risk_factors.extend(gdelt_risk.get("primary_concerns", []))
        
        return {
            "port_name": port_name,
            "total_risk_score": round(total_risk, 2),
            "risk_level": risk_level,
            "color": color,
            "risk_factors": risk_factors,
            "weather_contribution": weather_score,
            "gdelt_contribution": gdelt_score,
            "events_count": len(port_events),
            "events": port_events,
            "ai_analysis": ai_analysis,
            "executive_brief": ai_analysis.get("executive_brief", ""),
            "primary_driver": ai_analysis.get("primary_driver", "Unknown"),
            "recommended_action": ai_analysis.get("recommended_action", ""),
            "timestamp": datetime.now().isoformat()
        }
    
    def _basic_risk_analysis(self, port_name: str, weather_risk: Dict, gdelt_risk: Dict) -> Dict:
        """
        Basic risk analysis fallback when AI is unavailable
        
        Args:
            port_name: Port name
            weather_risk: Weather risk data
            gdelt_risk: GDELT risk data
            
        Returns:
            Basic risk analysis
        """
        weather_score = weather_risk.get("risk_score", 0)
        gdelt_score = gdelt_risk.get("risk_score", 0)
        total_risk = (weather_score * 0.4) + (gdelt_score * 0.6)
        
        if total_risk >= 70:
            risk_level = "Critical"
        elif total_risk >= 50:
            risk_level = "High"
        elif total_risk >= 30:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Determine primary driver
        if gdelt_score > weather_score:
            primary_driver = "GDELT"
            brief = f"Human-driven disruptions at {port_name}. "
            if gdelt_risk.get("primary_concerns"):
                brief += gdelt_risk["primary_concerns"][0]
        elif weather_score > gdelt_score:
            primary_driver = "OpenWeather"
            brief = f"Weather-related risks at {port_name}. "
            if weather_risk.get("factors"):
                brief += weather_risk["factors"][0]
        else:
            primary_driver = "Both"
            brief = f"Multiple risk factors detected at {port_name}"
        
        return {
            "location": port_name,
            "risk_level": risk_level,
            "primary_driver": primary_driver,
            "executive_brief": brief,
            "recommended_action": "Monitor situation and prepare contingency plans"
        }
    
    def generate_executive_summary(self, risk_data: Dict) -> str:
        """
        Generate an executive summary for a port's risk
        
        Args:
            risk_data: Comprehensive risk data
            
        Returns:
            Executive summary text
        """
        port = risk_data["port_name"]
        risk_level = risk_data["risk_level"].upper()
        risk_score = risk_data["total_risk_score"]
        
        summary = f"**{port} - {risk_level} RISK (Score: {risk_score}/100)**\n\n"
        
        # Add AI-generated executive brief
        if risk_data.get("executive_brief"):
            summary += f"**AI Analysis:** {risk_data['executive_brief']}\n\n"
        
        # Primary driver
        if risk_data.get("primary_driver"):
            summary += f"**Primary Driver:** {risk_data['primary_driver']}\n\n"
        
        # Events
        if risk_data["events_count"] > 0:
            summary += f"⚠️ **{risk_data['events_count']} Active Event(s):**\n"
            for event in risk_data["events"][:3]:  # Show top 3
                summary += f"- {event.get('event_description', 'Unknown event')}\n"
            summary += "\n"
        
        # Weather impact
        if risk_data["weather_contribution"] > 30:
            summary += f"🌤️ **Weather Impact:** {risk_data['weather_contribution']:.1f} points\n\n"
        
        # GDELT impact
        if risk_data["gdelt_contribution"] > 30:
            summary += f"📰 **Event Impact:** {risk_data['gdelt_contribution']:.1f} points\n\n"
        
        # Key risk factors
        summary += "**Key Risk Factors:**\n"
        for factor in risk_data["risk_factors"][:5]:  # Top 5 factors
            summary += f"- {factor}\n"
        
        # AI Recommendations
        summary += f"\n**Recommended Action:**\n{risk_data.get('recommended_action', 'Monitor situation closely')}\n"
        
        return summary
    
    def get_high_risk_ports(self, all_risks: List[Dict], threshold: int = 50) -> List[Dict]:
        """
        Filter ports with risk scores above threshold
        
        Args:
            all_risks: List of all port risk data
            threshold: Minimum risk score to include
            
        Returns:
            List of high-risk ports
        """
        return [
            risk for risk in all_risks 
            if risk["total_risk_score"] >= threshold
        ]


if __name__ == "__main__":
    # Test the risk analyzer
    analyzer = RiskAnalyzer()
    
    print("Testing Risk Analyzer with AI Integration...")
    print("\nFetching GDELT events...\n")
    
    events = analyzer.gdelt_service.get_all_events()
    for event in events[:3]:
        print(f"- {event['location']}: {event['event_description']}")
    
    print("\n" + "="*80 + "\n")
    
    # Test comprehensive risk calculation
    sample_weather_risk = {
        "risk_score": 35,
        "risk_level": "medium",
        "factors": ["Moderate wind speed: 12 m/s", "Reduced visibility: 3000m"]
    }
    
    risk = analyzer.calculate_comprehensive_risk("Rotterdam", sample_weather_risk, events)
    print(f"Comprehensive Risk Analysis for {risk['port_name']}:")
    print(f"Total Risk Score: {risk['total_risk_score']}")
    print(f"Risk Level: {risk['risk_level'].upper()}")
    print(f"Primary Driver: {risk['primary_driver']}")
    print(f"\nExecutive Summary:")
    print(analyzer.generate_executive_summary(risk))

# Made with Bob
