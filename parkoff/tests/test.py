

from parkoff.base import ParkOff


inst = ParkOff(settings_json="/Users/riteshtekriwal/Work/GitClones/parkoff/settings.json")


inst.cache_all_images()

inst.analyse_image()