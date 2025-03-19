from typing import Optional, List
import dspy


class GenerateReportSummary(dspy.Signature):
    """Generate a comprehensive executive summary of a report along with key recommendations.
    
    The summary should:
    1. Highlight the main findings and insights from the report
    2. Identify key areas of importance or concern
    3. Provide actionable recommendations based on the report content
    4. Consider the persona/role of the reader when making recommendations
    5. Format the output in markdown for readability
    
    Use a professional, concise writing style appropriate for an executive summary.
    """
    
    topic = dspy.InputField(prefix="Topic of the report:", format=str)
    report_content = dspy.InputField(prefix="Full content of the report:", format=str)
    personas = dspy.InputField(prefix="Personas/roles to consider for recommendations (leave blank if none):", format=str)
    summary = dspy.OutputField(
        prefix="Executive summary with key insights and recommendations in markdown format:",
        format=str
    )


class ReportSummaryModule(dspy.Module):
    """Generate an executive summary and recommendations for a report.
    
    This module analyzes the content of a report and produces:
    1. A concise summary of the key findings
    2. A list of actionable recommendations based on those findings
    3. Tailored recommendations for specific personas if provided
    
    The output is formatted in markdown for easy inclusion at the beginning of the report.
    """
    
    def __init__(
        self,
        engine: dspy.LM,
    ):
        super().__init__()
        self.generate_summary = dspy.Predict(GenerateReportSummary)
        self.engine = engine
    
    def forward(
        self, 
        topic: str, 
        report_content: str, 
        personas: Optional[List[str]] = None
    ) -> str:
        """
        Generate an executive summary with recommendations for a report.
        
        Args:
            topic: The main topic of the report
            report_content: The full content of the report
            personas: Optional list of personas to consider for targeted recommendations
            
        Returns:
            A markdown-formatted executive summary with recommendations
        """
        # Format personas if provided
        personas_text = ""
        if personas and len(personas) > 0:
            personas_text = ", ".join(personas)
        
        # Generate the summary with recommendations
        with dspy.settings.context(lm=self.engine):
            result = self.generate_summary(
                topic=topic,
                report_content=report_content,
                personas=personas_text
            )
            
        return result.summary
