#%%
def excited_decorator(func):
    def wrapper():
        # Add extra behavior before calling the original function
        result = func()
        # Modify the result
        return f"{result} I'm so excited!"
    return wrapper

#%%
@excited_decorator
def greet():
    return "Hello!"

#%%
greet()
# %%
