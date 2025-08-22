from database import add_dataset

# Création des tables
if __name__ == "__main__":
    # print("Création des tables...")
    # Base.metadata.create_all(bind=engine)
    add_dataset()
    print("OK")
