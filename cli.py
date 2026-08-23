from agent import Agent
from dotenv import load_dotenv
from eval_checks import detect_handoff
import sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


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
            if user_input.strip().lower() in ["exit", "quit"]:
                break

            response = agent.send_message(user_input)
            print(f"\nAgent: {response}")

            sources = agent.last_trace.get("sources_used") or []
            if sources:
                print("\n[Sources cited:]")
                for source in sorted(set(sources)):
                    print(f"  - {source}")

            if detect_handoff(response) or agent.last_trace.get("handoff"):
                print("\n[Flag: Handoff Recommended]")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
