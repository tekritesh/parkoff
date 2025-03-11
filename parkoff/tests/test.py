

from parkoff.base import ParkOff


inst = ParkOff(settings_json="settings.json")


inst.cache_all_images()

inst.analyse_image()