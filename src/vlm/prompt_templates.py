"""
Prompt Templates for VLM Copilot
==================================
Structured prompts for different inspection tasks.
"""

QUALITY_ASSESSMENT_PROMPT = """Analyze this steel surface image and provide a structured quality assessment:

1. **Defect Detection**: Are any surface defects visible? (Yes/No)
2. **Defect Types**: If defects are found, classify them:
   - Crazing (fine crack networks)
   - Inclusion (embedded particles)
   - Patches (surface discoloration)
   - Pitted Surface (small cavities)
   - Rolled-in Scale (oxide pressed in)
   - Scratches (linear marks)
3. **Severity**: Rate as Critical / High / Medium / Low / None
4. **Location**: Describe where defects are located in the image
5. **Recommendation**: Accept / Reject / Re-inspect

Provide a concise, factual assessment."""


DEFECT_CLASSIFICATION_PROMPT = """Classify the defect visible in this steel surface image.

Choose from these defect categories:
- Crazing: Fine network of hairline cracks
- Inclusion: Non-metallic particles embedded in steel
- Patches: Irregular surface discoloration
- Pitted Surface: Small pits or cavities
- Rolled-in Scale: Oxide scale pressed into surface
- Scratches: Linear marks from mechanical contact

Provide:
1. Primary defect type
2. Confidence level (High/Medium/Low)
3. Brief description of visible characteristics"""


SEVERITY_RATING_PROMPT = """Rate the severity of the surface defect in this image.

Severity Scale:
- **Critical**: Structural integrity compromised, immediate rejection
- **High**: Significant cosmetic defect, likely rejection
- **Medium**: Moderate defect, requires further inspection
- **Low**: Minor surface imperfection, may be acceptable
- **None**: No visible defects, surface acceptable

Provide:
1. Severity rating
2. Reasoning
3. Impact on product quality"""


COMPARISON_PROMPT = """Compare these two steel surface images and assess:

1. Which image shows more severe defects?
2. What types of defects are present in each?
3. Are the defects of the same type or different?
4. Quality ranking of each image (1-10 scale)

Provide a concise comparative analysis."""


DRONE_INSPECTION_PROMPT = """You are analyzing an aerial image captured by an inspection drone.

Identify:
1. Infrastructure or surface being inspected
2. Any visible defects, damage, or anomalies
3. Structural concerns
4. Recommended follow-up actions
5. Areas requiring closer inspection

Be specific about locations and severity."""


ROOT_CAUSE_PROMPT = """Analyze this steel surface defect and determine the likely root cause.

Consider manufacturing process factors:
- Rolling temperature and speed
- Cooling rate and method
- Raw material quality
- Equipment condition
- Environmental factors

Provide:
1. Most likely root cause
2. Contributing factors
3. Prevention recommendations"""
