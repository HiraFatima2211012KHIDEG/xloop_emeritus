from typing import List
class User:
    subscription: bool

def notify(user: User) -> None:
    pass

def get_subscribed_users(users: List[User]):
    """"Filter user that subscribed"""
    return [user for user in users if user.subscription]

def notify_subscribed_users(users: List[User]):
    """"Notify subscribed users"""
    for user in get_subscribed_users(users):
        notify(user)

