from agents.master_agent import create_master_agent

def main():
    print("🔬 Welcome! To search for papers, just type a topic. To ask a question about the papers, ask a follow-up after searching.")
    master_agent = create_master_agent()
    chat_history = []
    while True:
        user_query = input("\n💬 What would you like me to help you with? ").strip()
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("👋 Goodbye! Happy researching!")
            break
        if not user_query:
            continue
        response = master_agent.invoke({
            "input": user_query,
            "chat_history": chat_history
        })
        print(f"\n📝 Response:\n{response['output']}")
        chat_history.append((user_query, response['output']))

if __name__ == "__main__":
    main() 