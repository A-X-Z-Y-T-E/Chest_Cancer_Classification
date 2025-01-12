from src.Chest_Cancer_Classification import logger
from src.Chest_Cancer_Classification.utils.data_setup import get_data_loaders

data_dir = "Data"
batch_size = 32

try:
    logger.info("Setting up data loaders...")
    train_data_loader, test_data_loader, valid_data_loader, class_names = get_data_loaders(root_dir=data_dir, batch_size=batch_size)
    logger.info("Data loaders set up successfully.")
    logger.info(f"Classes: {class_names}")
    logger.info(f"Number of training samples: {len(train_data_loader.dataset)}")
    logger.info(f"Number of testing samples: {len(test_data_loader.dataset)}")
    logger.info(f"Number of validation samples: {len(valid_data_loader.dataset)}")
    
except Exception as e:
    logger.exception("An error occurred while setting up data loaders.")
    raise e