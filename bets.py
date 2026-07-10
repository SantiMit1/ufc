from src.predict import predict_event_valuebets

fights = [
    {"fighter_a": "Alessandro Costa", "fighter_b": "Cody Durden", "odds_a": 1.39, "odds_b": 3.03, "category": "Flyweight"},
    {"fighter_a": "Ryan Gandra", "fighter_b": "Zach Reese", "odds_a": 1.67, "odds_b": 2.2, "category": "Middleweight"},
    {"fighter_a": "Farid Basharat", "fighter_b": "John Garza", "odds_a": 1.16, "odds_b": 5.20, "category": "Bantamweight"},
    {"fighter_a": "Damian Pinas", "fighter_b": "Cesar Almeida", "odds_a": 1.37, "odds_b": 3.1, "category": "Middleweight"},
    {"fighter_a": "Tracy Cortez", "fighter_b": "Wang Cong", "odds_a": 1.98, "odds_b": 1.85, "category": "Women's Flyweight"},
    {"fighter_a": "Luke Riley", "fighter_b": "Kai Kamaka III", "odds_a": 1.35, "odds_b": 3.23, "category": "Featherweight"},
    {"fighter_a": "Cody Garbrandt", "fighter_b": "Adrian Yanez", "odds_a": 4.05, "odds_b": 1.25, "category": "Bantamweight"},
    {"fighter_a": "Gable Stevenson", "fighter_b": "Elisha Ellison", "odds_a": 1.05, "odds_b": 10.25, "category": "Heavyweight"},
    {"fighter_a": "Nikita Krylov", "fighter_b": "Robert Whittaker", "odds_a": 2.13, "odds_b": 1.72, "category": "Light Heavyweight"},
    {"fighter_a": "King Green", "fighter_b": "Terrance McKinney", "odds_a": 2.10, "odds_b": 1.75, "category": "Lightweight"},
    {"fighter_a": "Brandon Royval", "fighter_b": "Lone'er Kavanagh", "odds_a": 2.62, "odds_b": 1.5, "category": "Flyweight"},
    {"fighter_a": "Cory Sandhagen", "fighter_b": "Mario Bautista", "odds_a": 1.68, "odds_b": 2.20, "category": "Bantamweight"},
    {"fighter_a": "Benoit Saint Denis", "fighter_b": "Paddy Pimblett", "odds_a": 1.62, "odds_b": 2.30, "category": "Lightweight"},
    {"fighter_a": "Conor McGregor", "fighter_b": "Max Holloway", "odds_a": 2.77, "odds_b": 1.45, "category": "Welterweight"},
]

predict_event_valuebets(fights, bankroll=4000.0)