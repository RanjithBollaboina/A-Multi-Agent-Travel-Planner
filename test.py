from backend import run_travel_agent


user_input=input("enter travel request")

response=run_travel_agent(
    user_input=user_input,
    thread_id="test_user"
    )

print(response["answer"])