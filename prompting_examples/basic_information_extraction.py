from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'


groceries_system_prompt = """
    Your task is to extract to a list all the groceries with its quantities based on the provided recipe
    Make sure that groceries are in order of appearance
    """

groceries_config = types.GenerateContentConfig(
    system_instruction=groceries_system_prompt
)

recipe = """
    Step 1:
  Grind 3 garlic cloves, knob of fresh ginger, roughly chopped, 3 spring onions to a paste in a food processor.
  Add 2 tbsp of clear honey, juice from one orange, 1 tbsp of light soy sauce and 2 tbsp of vegetable oil, then blend again.
  Pour the mixture over the cubed chicken from 4 small breast fillets and leave to marnate for at least 1hr.
  Toss in the 20 button mushrooms for the last half an hour so the take on some of the flavour, too.

  Step 2:
  Thread the chicken, 20 cherry tomatoes, mushrooms and 2 large red peppers onto 20 wooden skewers,
  then cook on a griddle pan for 7-8 mins each side or until the chicken is thoroughly cooked and golden brown.
  Turn the kebabs frequently and baste with the marinade from time to time until evenly cooked.
  Arrange on a platter, and eat with your fingers.
    """

# prompt = 'Explain how AI works in a few words'

response = client.models.generate_content(
    model=MODEL_ID,
    contents=recipe,
    config=groceries_config
)

print(response.text)