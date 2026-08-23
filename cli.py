from agent import Agent
from dotenv import load_dotenv
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    load_dotenv()
    print("Welcome to Aster & Row Support. Type 'exit' or 'quit' to stop.")
    
    try:
        agent = Agent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure GROQ_API_KEY and GEMINI_API_KEY are set in your environment or .env file.")
        return
        
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
                
            response = agent.send_message(user_input)
            print(f"\nAgent: {response}")
            
            trace = agent.last_trace
            if trace.get("sources_used"):
                print("\n[Sources cited:]")
                for s in sorted(list(set(trace["sources_used"]))):
                    print(f"  - {s}")
                    
            handoff_keywords = ["human", "support", "agent", "representative", "contact", "team"]
            if any(k in response.lower() for k in handoff_keywords):
                print("\n[Flag: Handoff Recommended]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
