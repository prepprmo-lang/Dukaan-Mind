import math
import random
from datetime import date, timedelta
from typing import List, Optional, Tuple


INDIAN_GROCERY_CATALOG = [
    ("DG-RICE-SONA-5K", "Sona Masoori Rice 5 kg", "Staples", 265.0, 315.0, 5.0),
    ("DG-RICE-BASMATI-1K", "Basmati Rice 1 kg", "Staples", 118.0, 149.0, 2.0),
    ("DG-RICE-BASMATI-5K", "Basmati Rice 5 kg", "Staples", 565.0, 675.0, 2.0),
    ("DG-RICE-IDLI-5K", "Idli Rice 5 kg", "Staples", 220.0, 275.0, 3.0),
    ("DG-RICE-BOILED-5K", "Boiled Rice 5 kg", "Staples", 230.0, 285.0, 3.0),
    ("DG-ATTA-5K", "Chakki Atta 5 kg", "Staples", 185.0, 225.0, 5.0),
    ("DG-ATTA-10K", "Chakki Atta 10 kg", "Staples", 355.0, 425.0, 2.0),
    ("DG-MAIDA-1K", "Maida 1 kg", "Staples", 37.0, 49.0, 2.0),
    ("DG-BESAN-1K", "Besan 1 kg", "Staples", 82.0, 105.0, 3.0),
    ("DG-SUJI-1K", "Sooji Rava 1 kg", "Staples", 41.0, 54.0, 2.0),
    ("DG-POHA-1K", "Thick Poha 1 kg", "Staples", 57.0, 74.0, 3.0),
    ("DG-DALIA-1K", "Broken Wheat Dalia 1 kg", "Staples", 46.0, 61.0, 2.0),
    ("DG-RAGI-1K", "Ragi Flour 1 kg", "Staples", 52.0, 69.0, 1.5),
    ("DG-VERMICELLI-400G", "Roasted Vermicelli 400 g", "Staples", 28.0, 37.0, 2.0),
    ("DG-SABUDANA-500G", "Sabudana 500 g", "Staples", 39.0, 52.0, 1.5),
    ("DG-TOOR-DAL-1K", "Toor Dal 1 kg", "Pulses", 142.0, 169.0, 5.0),
    ("DG-MOONG-DAL-1K", "Moong Dal 1 kg", "Pulses", 108.0, 132.0, 3.0),
    ("DG-MOONG-WHOLE-1K", "Whole Green Moong 1 kg", "Pulses", 94.0, 118.0, 2.0),
    ("DG-CHANADAL-1K", "Chana Dal 1 kg", "Pulses", 76.0, 96.0, 3.0),
    ("DG-MASOOR-DAL-1K", "Masoor Dal 1 kg", "Pulses", 91.0, 114.0, 2.5),
    ("DG-URAD-DAL-1K", "Urad Dal 1 kg", "Pulses", 119.0, 146.0, 2.0),
    ("DG-RAJMA-1K", "Kashmiri Rajma 1 kg", "Pulses", 132.0, 165.0, 1.5),
    ("DG-CHICKPEA-1K", "Kabuli Chana 1 kg", "Pulses", 96.0, 122.0, 2.0),
    ("DG-KALA-CHANA-1K", "Kala Chana 1 kg", "Pulses", 77.0, 98.0, 2.0),
    ("DG-SOYA-CHUNKS-500G", "Soya Chunks 500 g", "Pulses", 58.0, 75.0, 2.0),
    ("DG-SUNFLOWER-OIL-1L", "Sunflower Oil 1 L", "Cooking Oil", 112.0, 136.0, 5.0),
    ("DG-GROUNDNUT-OIL-1L", "Groundnut Oil 1 L", "Cooking Oil", 158.0, 189.0, 2.0),
    ("DG-MUSTARD-OIL-1L", "Mustard Oil 1 L", "Cooking Oil", 126.0, 154.0, 3.0),
    ("DG-COCONUT-OIL-500", "Coconut Oil 500 ml", "Cooking Oil", 82.0, 105.0, 1.5),
    ("DG-GHEE-500", "Cow Ghee 500 ml", "Cooking Oil", 286.0, 335.0, 1.0),
    ("DG-SUGAR-1K", "Sugar 1 kg", "Essentials", 39.0, 47.0, 8.0),
    ("DG-JAGGERY-1K", "Jaggery 1 kg", "Essentials", 55.0, 72.0, 2.0),
    ("DG-SALT-1K", "Iodized Salt 1 kg", "Essentials", 17.0, 22.0, 6.0),
    ("DG-TURMERIC-200", "Turmeric Powder 200 g", "Spices", 39.0, 52.0, 2.0),
    ("DG-CHILLI-200", "Red Chilli Powder 200 g", "Spices", 52.0, 69.0, 2.0),
    ("DG-CORIANDER-200", "Coriander Powder 200 g", "Spices", 34.0, 46.0, 2.0),
    ("DG-CUMIN-200", "Cumin Seeds 200 g", "Spices", 61.0, 79.0, 1.5),
    ("DG-MUSTARD-SEED-200", "Mustard Seeds 200 g", "Spices", 28.0, 39.0, 1.5),
    ("DG-GARAM-MASALA-100", "Garam Masala 100 g", "Spices", 48.0, 65.0, 1.5),
    ("DG-PEPPER-100", "Black Pepper 100 g", "Spices", 76.0, 98.0, 1.0),
    ("DG-CARDAMOM-50", "Green Cardamom 50 g", "Spices", 108.0, 139.0, 0.7),
    ("DG-TEA-250", "Assam Tea 250 g", "Beverages", 82.0, 105.0, 3.0),
    ("DG-COFFEE-100", "Instant Coffee 100 g", "Beverages", 118.0, 149.0, 1.5),
    ("DG-MILK-500", "Toned Milk 500 ml", "Dairy & Bakery", 25.0, 30.0, 12.0),
    ("DG-CURD-400", "Fresh Curd 400 g", "Dairy & Bakery", 34.0, 42.0, 5.0),
    ("DG-PANEER-200", "Paneer 200 g", "Dairy & Bakery", 72.0, 89.0, 2.0),
    ("DG-BREAD-400", "White Bread 400 g", "Dairy & Bakery", 29.0, 36.0, 5.0),
    ("DG-EGGS-6", "Farm Eggs 6 pack", "Dairy & Bakery", 38.0, 48.0, 4.0),
    ("DG-BUTTER-100", "Butter 100 g", "Dairy & Bakery", 54.0, 66.0, 2.0),
    ("DG-PAPAD-200", "Punjabi Papad 200 g", "Snacks", 44.0, 59.0, 1.5),
    ("DG-MIXTURE-200", "South Indian Mixture 200 g", "Snacks", 49.0, 65.0, 2.0),
    ("DG-CHIPS-100", "Masala Potato Chips 100 g", "Snacks", 27.0, 35.0, 5.0),
    ("DG-BISCUIT-250", "Glucose Biscuits 250 g", "Snacks", 23.0, 30.0, 6.0),
    ("DG-COOKIES-200", "Cashew Cookies 200 g", "Snacks", 42.0, 55.0, 2.0),
    ("DG-NOODLES-4", "Masala Noodles 4 pack", "Snacks", 48.0, 60.0, 5.0),
    ("DG-PEANUT-200", "Roasted Peanuts 200 g", "Snacks", 38.0, 52.0, 2.0),
    ("DG-RUSK-200", "Elaichi Rusk 200 g", "Snacks", 31.0, 42.0, 2.0),
    ("DG-COLA-750", "Cola 750 ml", "Beverages", 35.0, 45.0, 4.0),
    ("DG-JUICE-1L", "Mango Drink 1 L", "Beverages", 78.0, 99.0, 2.0),
    ("DG-BOTTLED-WATER-1L", "Packaged Water 1 L", "Beverages", 14.0, 20.0, 8.0),
    ("DG-SOAP-BAR", "Neem Bath Soap 100 g", "Personal Care", 31.0, 42.0, 3.0),
    ("DG-SHAMPOO-180", "Shampoo 180 ml", "Personal Care", 86.0, 109.0, 1.5),
    ("DG-TOOTHPASTE-150", "Mint Toothpaste 150 g", "Personal Care", 67.0, 85.0, 2.0),
    ("DG-TOOTHBRUSH", "Soft Toothbrush", "Personal Care", 28.0, 39.0, 1.5),
    ("DG-HAIR-OIL-200", "Coconut Hair Oil 200 ml", "Personal Care", 54.0, 72.0, 1.5),
    ("DG-DETERGENT-1K", "Laundry Detergent 1 kg", "Home Care", 78.0, 99.0, 3.0),
    ("DG-DISHWASH-500", "Dishwash Liquid 500 ml", "Home Care", 61.0, 79.0, 2.0),
    ("DG-FLOOR-CLEANER-1L", "Floor Cleaner 1 L", "Home Care", 89.0, 115.0, 1.0),
    ("DG-INCENSE-100", "Incense Sticks 100 g", "Home Care", 34.0, 49.0, 2.0),
    ("DG-LED-BATTERIES", "AA Batteries 2 pack", "Home Care", 39.0, 55.0, 1.0),
]


def generate_indian_grocery_demo(
    days: int = 180,
    seed: int = 2026,
    *,
    start_date: Optional[date] = None,
    weekly_variation: bool = True,
) -> Tuple[List[tuple], List[tuple]]:
    if days < 2:
        raise ValueError("Demo sales history requires at least two days.")

    rng = random.Random(seed)
    products = []
    sales = []
    first_date = start_date or date.today() - timedelta(days=days)

    weekly_traffic = [1.0] * math.ceil(days / 7)
    if weekly_variation:
        traffic_rng = random.Random(seed + 7919)
        traffic = 1.0
        for week_index in range(len(weekly_traffic)):
            broad_cycle = (
                1.0
                + 0.1 * math.sin(2 * math.pi * week_index / 13)
                + 0.04 * math.sin(2 * math.pi * week_index / 5)
            )
            traffic = 0.55 * traffic + 0.45 * broad_cycle + traffic_rng.gauss(0, 0.18)
            weekly_traffic[week_index] = max(0.65, min(1.4, traffic))

    for sku, name, category, cost, price, average_daily_sales in INDIAN_GROCERY_CATALOG:
        stock = max(8, round(average_daily_sales * rng.uniform(4, 9)))
        reorder_level = max(4, round(average_daily_sales * 2.5))
        products.append((sku, name, category, stock, reorder_level, price, cost))

        for day_offset in range(days):
            sale_date = first_date + timedelta(days=day_offset)
            sale_probability = min(0.95, 0.45 + average_daily_sales / 20)
            if rng.random() > sale_probability:
                continue

            weekend_factor = 1.18 if sale_date.weekday() >= 5 else 0.96
            seasonal_factor = 1.0 + 0.08 * (
                1 + math.sin((sale_date.timetuple().tm_yday / 365) * 2 * math.pi)
            )
            promotion_factor = 1.3 if rng.random() < 0.035 else 1.0
            expected_quantity = (
                average_daily_sales
                * weekend_factor
                * seasonal_factor
                * promotion_factor
                * weekly_traffic[day_offset // 7]
            )
            quantity = max(1, round(rng.gauss(expected_quantity, max(0.5, expected_quantity * 0.24))))
            sales.append(
                (
                    sale_date.isoformat(),
                    sku,
                    name,
                    quantity,
                    round(quantity * price, 2),
                    cost,
                )
            )

    return products, sales
