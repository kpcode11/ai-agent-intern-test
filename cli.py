from agent import Agent
from dotenv import load_dotenv

def main():
    load_dotenv()
    print("Welcome to Aster & Row Support. Type 'exit' or 'quit' to stop.")
    
    try:
        agent = Agent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure you have set GEMINI_API_KEY in your environment or .env file.")
        return
        
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
                
            response = agent.send_message(user_input)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
