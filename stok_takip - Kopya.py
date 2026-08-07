import requests
from bs4 import BeautifulSoup
import psycopg2
import time
import re
import json
import os
import random
from urllib.parse import urljoin
from datetime import datetime, timedelta
import subprocess

class ArabamStokTakip:
    def __init__(self, db_config=None):
        self.base_url = "https://www.arabam.com"

        # ═══ DB Ayarları ═══
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'araba_ilan',
            'user': 'postgres',
            'password': 'sifre123'
        }
        self.db_conn = None
        self.db_cursor = None
        self.scrape_log_id = None

        # ═══ Bugün sitede görülen ilanlar ═══
        # Liste sayfalarından toplanan: {ilan_no: {'fiyat': X, 'url': Y}}
        self.bugun_gorulen = {}

        # Yeni ilan URL'leri (DB'de olmayan, detay çekilecek)
        self.yeni_ilan_urls = []

        # DB'den yüklenen aktif ilanlar: {kaynak_ilan_no: (id, fiyat)}
        self.db_aktif = {}

        # ═══ Durum kayıt dosyası (yarıda kalırsa devam için) ═══
        self.state_file = 'stok_takip_state.json'

        # ═══ İstatistikler ═══
        self.stats = {
            'taranan_sayfa': 0,
            'taranan_ilan': 0,
            'yeni_ilan': 0,
            'fiyat_degisen': 0,
            'son_gorulme_guncellenen': 0,
            'satilan': 0,
            'hatali': 0,
            'toplam_istek': 0,
            'baslangic': None,
        }

        self.failed_urls = []

        # ═══ Anti-Ban: User-Agent Havuzu ═══
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) '
            'Gecko/20100101 Firefox/126.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
            '(KHTML, like Gecko) Version/17.5 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
        ]

        self.referrers = [
            'https://www.google.com/',
            'https://www.google.com.tr/',
            'https://www.arabam.com/',
            'https://www.arabam.com/ikinci-el/otomobil',
            None,
        ]

        # Anti-ban limitleri
        self.request_count = 0
        self.MAX_REQUESTS_PAUSE = random.randint(350, 500)      # Daha geç mola
        self.MAX_REQUESTS_REFRESH = random.randint(1200, 1800)
        # ═══ 13 Parça eşleştirme (Part 3'te kullanılacak) ═══
        self.part_columns = {
            'Sağ Arka Çamurluk': 'sag_arka_camurluk',
            'Arka Kaput': 'arka_kaput',
            'Sol Arka Çamurluk': 'sol_arka_camurluk',
            'Sağ Arka Kapı': 'sag_arka_kapi',
            'Sağ Ön Kapı': 'sag_on_kapi',
            'Tavan': 'tavan',
            'Sol Arka Kapı': 'sol_arka_kapi',
            'Sol Ön Kapı': 'sol_on_kapi',
            'Sağ Ön Çamurluk': 'sag_on_camurluk',
            'Motor Kaputu': 'motor_kaputu',
            'Sol Ön Çamurluk': 'sol_on_camurluk',
            'Ön Tampon': 'on_tampon',
            'Arka Tampon': 'arka_tampon',
        }

                # ═══ TARANACAK KATEGORİLER ═══
        self.categories = [
            {
                'name': 'Otomobil',
                'slug': 'otomobil',
                'path': '/ikinci-el/otomobil',
            },
            {
                'name': 'Arazi, SUV, Pick-up',
                'slug': 'arazi-suv-pick-up',          # ← DİKKAT: pick-up
                'path': '/ikinci-el/arazi-suv-pick-up',
            },
        ]

        # ═══ Tire içeren marka slug'ları ═══
        self.multi_word_brands = {
            'mercedes-benz', 'alfa-romeo', 'rolls-royce',
            'aston-martin', 'land-rover', 'ds-automobiles',
            'de-tomaso', 'great-wall',
        }

        # Session başlat
        self.session = None
        self._create_session()

    # ══════════════════════════════════════════════════════════════
    #  VERİTABANI İŞLEMLERİ
    # ══════════════════════════════════════════════════════════════
    def db_connect(self, retry=True):
        """PostgreSQL bağlantısı kur, kapalıysa bat dosyasıyla başlatmayı dene"""
        try:
            self.db_conn = psycopg2.connect(**self.db_config)
            self.db_conn.autocommit = False
            self.db_cursor = self.db_conn.cursor()
            print("✅ PostgreSQL bağlantısı kuruldu")
            return True
        except Exception as e:
            if retry:
                print("\n  🔄 Veritabanı kapalı görünüyor. 'baslat.bat' tetikleniyor...")
                import subprocess
                import time
                bat_path = r'"C:\pgsql\postgresql-17.9-2-windows-x64-binaries\pgsql\pgAdmin 4\runtime\baslat.bat"'
                try:
                    subprocess.run(bat_path, shell=True, capture_output=True)
                    print("  ⏳ DB'nin açılması için 5 saniye bekleniyor...")
                    time.sleep(5)
                    # Sadece 1 kere tekrar dene
                    return self.db_connect(retry=False)  
                except Exception as ex:
                    print(f"  ❌ Bat dosyası çalıştırılamadı: {ex}")
            else:
                print(f"❌ DB bağlantı hatası: {e}")
            
            return False

    def check_db_connection(self):
        """DB bağlantısı koptuysa yeniden bağlan"""
        try:
            if self.db_conn is None or self.db_cursor is None:
                raise psycopg2.InterfaceError("Bağlantı yok")
            self.db_cursor.execute("SELECT 1")
            return True
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            print("\n  🔄 DB bağlantısı kopmuş. Yeniden bağlanılıyor...")
            return self.db_connect(retry=False)
        except Exception:
            return self.db_connect(retry=False)

    def db_disconnect(self):
        """Bağlantıyı kapat"""
        try:
            if self.db_cursor:
                self.db_cursor.close()
            if self.db_conn:
                self.db_conn.close()
            print("🔌 DB bağlantısı kapatıldı")
        except:
            pass

    def db_load_aktif_ilanlar(self):
        """
        DB'deki TÜM aktif arabam ilanlarını yükle.
        Return: {kaynak_ilan_no: (id, fiyat)}
        """
        try:
            self.db_cursor.execute("""
                SELECT id, kaynak_ilan_no, fiyat
                FROM ilanlar
                WHERE kaynak = 'arabam' AND ilan_durumu = 'aktif'
            """)
            rows = self.db_cursor.fetchall()
            self.db_aktif = {}
            for row in rows:
                db_id = row[0]
                ilan_no = str(row[1])
                fiyat = float(row[2]) if row[2] else None
                self.db_aktif[ilan_no] = (db_id, fiyat)

            print(f"📊 DB'de {len(self.db_aktif):,} aktif ilan yüklendi")
            return len(self.db_aktif)
        except Exception as e:
            print(f"⚠️ DB aktif ilan yükleme hatası: {e}")
            self.db_conn.rollback()
            return 0

    def db_log_start(self):
        """Scrape başlangıç logu"""
        try:
            self.db_cursor.execute("""
                INSERT INTO scrape_log (kaynak, durum, notlar)
                VALUES ('arabam', 'devam_ediyor', 'Günlük stok takip başladı')
                RETURNING id
            """)
            self.scrape_log_id = self.db_cursor.fetchone()[0]
            self.db_conn.commit()
        except:
            self.db_conn.rollback()
            self.scrape_log_id = None

    def db_log_end(self, durum='basarili'):
        """Scrape bitiş logu"""
        if not self.scrape_log_id:
            return
        try:
            s = self.stats
            self.db_cursor.execute("""
                UPDATE scrape_log SET
                    bitis_zamani = NOW(),
                    toplam_ilan = %s,
                    yeni_ilan = %s,
                    guncellenen_ilan = %s,
                    satilan_ilan = %s,
                    hata_sayisi = %s,
                    durum = %s,
                    notlar = %s
                WHERE id = %s
            """, (
                s['taranan_ilan'],
                s['yeni_ilan'],
                s['fiyat_degisen'] + s['son_gorulme_guncellenen'],
                s['satilan'],
                s['hatali'],
                durum,
                json.dumps(s, ensure_ascii=False, default=str),
                self.scrape_log_id
            ))
            self.db_conn.commit()
        except:
            self.db_conn.rollback()

    # ══════════════════════════════════════════════════════════════
    #  SESSION + ANTI-BAN
    # ══════════════════════════════════════════════════════════════

    def _create_session(self):
        """Yeni session oluştur"""
        if self.session:
            try:
                self.session.close()
            except:
                pass

        self.session = requests.Session()
        ua = random.choice(self.user_agents)
        self.session.headers.update({
            'User-Agent': ua,
            'Accept': ('text/html,application/xhtml+xml,application/xml;'
                    'q=0.9,image/webp,*/*;q=0.8'),
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Cache-Control': 'max-age=0',
        })
        self.request_count = 0

    def _refresh_session(self):
        """Session'ı yeni UA ile sıfırla"""
        self._create_session()
        self.MAX_REQUESTS_PAUSE = random.randint(350, 500) #eski 200-350
        print("    🔄 Session yenilendi (yeni User-Agent)")

    def _smart_delay(self, request_type='listing'):
        """Anti-ban - AGRESİF MOD (%50 hızlandırılmış)"""
        self.request_count += 1
        self.stats['toplam_istek'] += 1

        # Büyük mola: session yenile
        if self.request_count >= self.MAX_REQUESTS_REFRESH:
            mola = random.uniform(8, 15)           # ESKİ: 15-30
            print(f"\n    🛑 {self.request_count} istek → "
                f"session yenileniyor + {mola:.0f}sn")
            self._refresh_session()
            time.sleep(mola)
            return

        # Orta mola
        if self.request_count % self.MAX_REQUESTS_PAUSE == 0:
            mola = random.uniform(3, 7)            # ESKİ: 7-15
            print(f"\n    ☕ {self.request_count} istek → {mola:.0f}sn mola")
            time.sleep(mola)
            return

        # Normal bekleme
        if request_type == 'listing':
            delay = random.uniform(0.10, 0.35)     # ESKİ: 0.25-0.75
        elif request_type == 'detail':
            delay = random.uniform(0.15, 0.35)     # ESKİ: 0.30-0.65
        else:
            delay = random.uniform(0.15, 0.35)     # ESKİ: 0.30-0.65

        # İnsan taklidi uzun bekleme
        if random.random() < 0.008:                # ESKİ: %1.5 → %0.8
            delay += random.uniform(0.8, 2.0)      # ESKİ: 1.5-3.5

        time.sleep(delay)
        
    def _get_random_headers(self):
        """Her istek için rastgele ek header"""
        headers = {}
        ref = random.choice(self.referrers)
        if ref:
            headers['Referer'] = ref
        if random.random() < 0.3:
            headers['Accept'] = ('text/html,application/xhtml+xml,'
                                'application/xml;q=0.9,*/*;q=0.8')
        return headers

    # ══════════════════════════════════════════════════════════════
    #  İNTERNET BAĞLANTI KONTROLÜ
    # ══════════════════════════════════════════════════════════════

    def wait_for_connection(self, max_wait=7200):
        """İnternet kesilirse geri gelene kadar bekle (maks 2 saat)"""
        test_urls = [
            "https://www.google.com",
            "https://www.arabam.com",
            "https://www.cloudflare.com",
        ]
        wait_time = 5
        total_waited = 0
        first_fail = True

        while total_waited < max_wait:
            for url in test_urls:
                try:
                    requests.get(url, timeout=10)
                    if total_waited > 0:
                        print(f"\n    ✅ İnternet geldi! "
                            f"({total_waited // 60}dk beklendi)")
                        self._refresh_session()
                        time.sleep(3)
                    return True
                except:
                    continue

            if first_fail:
                print(f"\n    ⚠️ İNTERNET KESİLDİ! Bekleniyor...")
                self.save_state()
                first_fail = False

            time.sleep(wait_time)
            total_waited += int(wait_time)

            if total_waited % 60 == 0:
                print(f"    ⏳ {total_waited // 60} dk bekleniyor...")

            wait_time = min(wait_time * 1.3, 60)

        return False

    # ══════════════════════════════════════════════════════════════
    #  SAYFA ÇEKME (DAYANIKLI)
    # ══════════════════════════════════════════════════════════════

    def get_page(self, url, retries=5, request_type='listing'):
        """Sayfayı çek - anti-ban + internet dayanıklı"""
        self._smart_delay(request_type)

        for attempt in range(retries):
            try:
                extra = self._get_random_headers()
                resp = self.session.get(url, timeout=30, headers=extra)

                if resp.status_code == 200:
                    return BeautifulSoup(resp.text, 'html.parser')

                elif resp.status_code == 429:
                    wait = random.uniform(30, 60) * (attempt + 1) # eski 60-120
                    print(f"    🚫 Rate limit! {wait:.0f}sn...")
                    time.sleep(wait)
                    self._refresh_session()

                elif resp.status_code == 403:
                    wait = random.uniform(60, 350) # eski 120-300
                    print(f"    🚫 403 Engel! {wait:.0f}sn + session yenile")
                    self._refresh_session()
                    time.sleep(wait)

                elif resp.status_code >= 500:
                    time.sleep(random.uniform(5, 15) * (attempt + 1)) # eski 10-30

                else:
                    time.sleep(random.uniform(3, 8))

            except (requests.ConnectionError, requests.Timeout,
                    ConnectionResetError, ConnectionAbortedError):
                if not self.wait_for_connection():
                    return None
                continue

            except Exception as e:
                print(f"    ⚠️ {type(e).__name__}: {e}")
                time.sleep(random.uniform(5, 15))

        self.failed_urls.append(url)
        return None

    # ══════════════════════════════════════════════════════════════
    #  YARDIMCI FONKSİYONLAR
    # ══════════════════════════════════════════════════════════════

    def extract_ilan_no_from_url(self, url):
        """URL'den ilan numarası çıkar"""
        match = re.search(r'/(\d{6,})(?:[/?#]|$)', url)
        if match:
            return match.group(1)
        parts = url.rstrip('/').split('/')
        for part in reversed(parts):
            clean = part.split('?')[0].split('#')[0]
            if clean.isdigit() and len(clean) >= 5:
                return clean
        return None



    def _is_brand_slug(self, slug):
        """
        Slug marka mı yoksa model mi?
        'ford' → True,  'mercedes-benz' → True,  'ford-fiesta' → False
        """
        if not slug:
            return False
        if slug in self.multi_word_brands:
            return True
        if '-' not in slug:
            return True
        return False

    def extract_ilan_no_from_page(self, soup):
        """Detay sayfasından ilan numarası çıkar"""
        for sel in ['.classified-id', '[class*="ilan-no"]']:
            elem = soup.select_one(sel)
            if elem:
                m = re.search(r'(\d{5,})', elem.get_text(strip=True))
                if m:
                    return m.group(1)
        for script in soup.select('script'):
            if script.string:
                for pat in [r'"classifiedId"\s*:\s*(\d+)',
                            r'"adId"\s*:\s*(\d+)']:
                    m = re.search(pat, script.string)
                    if m:
                        return m.group(1)
        return None

    def parse_fiyat(self, text):
        """'355.000 TL' → 355000.0"""
        if not text:
            return None
        text = str(text).replace('TL', '').replace('tl', '')
        text = text.replace('.', '').replace(',', '.').strip()
        try:
            return float(text)
        except:
            return None

    def parse_km(self, text):
        """'201.000' → 201000"""
        if not text:
            return None
        text = str(text).replace('km', '').replace('KM', '')
        text = text.replace('.', '').replace(',', '').strip()
        try:
            return int(float(text))
        except:
            return None

    def parse_yil(self, text):
        """'1998' → 1998"""
        if not text:
            return None
        try:
            yil = int(float(str(text).strip()))
            return yil if 1950 <= yil <= 2030 else None
        except:
            return None

    def parse_tarih(self, text):
        """'18 Mart 2026' → date(2026, 3, 18)"""
        if not text:
            return None
        try:
            ay_map = {
                'ocak': '01', 'şubat': '02', 'mart': '03',
                'nisan': '04', 'mayıs': '05', 'haziran': '06',
                'temmuz': '07', 'ağustos': '08', 'eylül': '09',
                'ekim': '10', 'kasım': '11', 'aralık': '12',
            }
            parts = str(text).strip().split()
            if len(parts) == 3:
                gun = parts[0].zfill(2)
                ay = ay_map.get(parts[1].lower())
                if ay:
                    return datetime.strptime(
                        f"{parts[2]}-{ay}-{gun}", "%Y-%m-%d"
                    ).date()
        except:
            pass
        return None

    def temizle_parca(self, val):
        """Parça durumu temizle → DB enum'a uygun hale getir"""
        parca_map = {
            'orijinal': 'orijinal',
            'boyali': 'boyali',
            'lokal_boyali': 'lokal_boyali',
            'degismis': 'degismis',
            'belirtilmemis': 'belirtilmemis',
            'bos': 'belirtilmemis',
        }
        if not val:
            return 'belirtilmemis'
        return parca_map.get(str(val).strip().lower(), 'belirtilmemis')

    def temizle_tramer(self, val):
        """Tramer tutarı temizle → float veya None"""
        if not val or str(val).strip() in ['-', '', '0']:
            return None
        val = str(val).replace('.', '').replace(',', '.')
        val = val.replace('TL', '').replace('tl', '').strip()
        try:
            result = float(val)
            return result if result >= 0 else None
        except:
            return None

    def get_total_pages(self, soup):
        """Toplam sayfa sayısı"""
        if soup:
            elem = soup.select_one('#js-hook-for-total-page-count')
            if elem:
                try:
                    return min(int(elem.get_text(strip=True)), 50)
                except:
                    pass
        return 1

    # ══════════════════════════════════════════════════════════════
    #  DURUM KAYDET / YÜKLE (Yarıda kalırsa devam)
    # ══════════════════════════════════════════════════════════════

    def save_state(self, phase='', brand_idx=0, model_idx=0,
                sub_idx=0, year_idx=0, cat_idx=0):
        state = {
            'phase': phase,
            'cat_idx': cat_idx,            # ← YENİ
            'brand_idx': brand_idx,
            'model_idx': model_idx,
            'sub_idx': sub_idx,
            'year_idx': year_idx,
            'bugun_gorulen': self.bugun_gorulen,
            'yeni_ilan_urls': self.yeni_ilan_urls,
            'bugun_gorulen_count': len(self.bugun_gorulen),
            'yeni_ilan_count': len(self.yeni_ilan_urls),
            'stats': self.stats,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except:
            pass

    def load_state(self):
        """Kaydedilmiş durumu yükle"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                # Sadece bugün kaydedildiyse geçerli
                ts = state.get('timestamp', '')
                if ts.startswith(datetime.now().strftime('%Y-%m-%d')):
                    print(f"📂 Bugünkü kayıt bulundu: {ts}")
                    print(f"   Görülen: {state.get('bugun_gorulen_count', 0)}")
                    return state
                else:
                    print(f"📂 Eski kayıt ({ts}), sıfırdan başlanacak")
                    os.remove(self.state_file)
            except:
                pass
        return None

    def clear_state(self):
        """State dosyasını sil"""
        try:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
        except:
            pass
    
    def print_stats(self, prefix=''):
        """Anlık istatistikleri yazdır"""
        s = self.stats
        elapsed = ''
        if s['baslangic']:
            sec = (datetime.now() - datetime.strptime(
            s['baslangic'], '%Y-%m-%d %H:%M:%S'
            )).total_seconds()
            elapsed = f" | Süre: {int(sec//3600)}sa {int((sec%3600)//60)}dk"
            print(f"{prefix}📊 Sayfa: {s['taranan_sayfa']:,} | "
            f"İlan: {s['taranan_ilan']:,} | "
            f"Yeni: {s['yeni_ilan']:,} | "
            f"Fiyat↕: {s['fiyat_degisen']:,} | "
            f"Satılan: {s['satilan']:,} | "
            f"İstek: {s['toplam_istek']:,}{elapsed}")
        
    # ══════════════════════════════════════════════════════════════
#  PART 2/4: LİSTE SAYFASI TARAMA
#
#  Tüm liste sayfalarını gezer, her satırdan:
#    ● ilan_no  → <tr id="listing38839199"> 'dan
#    ● fiyat    → .listing-price 'dan
#    ● url      → a[href*="/ilan/"] 'dan
#
#  Detay sayfasına GİRMEZ → çok hızlı (~3000 sayfa/saat)
# ══════════════════════════════════════════════════════════════

    def parse_listing_row(self, row):
        """
        Tek bir <tr> satırından ilan_no + fiyat + url çıkar.

        <tr id="listing38839199" class="listing-list-item ...">
        ...
        <span class="db no-wrap listing-price">355.000 TL</span>
        ...
        <a href="/ilan/.../38839199">
        """
        result = None
        try:
            # ═══ İLAN NO: <tr id="listing38839199"> ═══
            tr_id = row.get('id', '')
            ilan_no = tr_id.replace('listing', '').strip()
            if not ilan_no or not ilan_no.isdigit():
                return None

            # ═══ FİYAT: <span class="listing-price">355.000 TL</span> ═══
            fiyat_elem = row.select_one('.listing-price')
            fiyat = None
            if fiyat_elem:
                fiyat = self.parse_fiyat(fiyat_elem.get_text(strip=True))

            # ═══ URL: a[href*="/ilan/"] ═══
            link = row.select_one('a[href*="/ilan/"]')
            url = None
            if link:
                href = link.get('href', '')
                url = urljoin(self.base_url, href)

            result = {
                'ilan_no': ilan_no,
                'fiyat': fiyat,
                'url': url,
            }

        except Exception as e:
            pass

        return result

    def scan_list_page(self, page_url):
        """
        Tek bir liste sayfasından TÜM ilanları çek (detaya GİRMEZ).
        Return: [{ilan_no, fiyat, url}, ...]
        """
        soup = self.get_page(page_url, request_type='listing')
        items = []

        if not soup:
            return items

        rows = soup.select('tr.listing-list-item')
        for row in rows:
            parsed = self.parse_listing_row(row)
            if parsed and parsed['ilan_no']:
                items.append(parsed)

        return items

    def scan_list_page_with_total(self, page_url):
        """
        İlk sayfayı çek + toplam sayfa sayısını döndür.
        Return: (items_list, total_pages)
        """
        soup = self.get_page(page_url, request_type='listing')
        items = []
        total_pages = 1

        if not soup:
            return items, total_pages

        total_pages = self.get_total_pages(soup)

        rows = soup.select('tr.listing-list-item')
        for row in rows:
            parsed = self.parse_listing_row(row)
            if parsed and parsed['ilan_no']:
                items.append(parsed)

        return items, total_pages

    # ══════════════════════════════════════════════════════════════
    #  MARKA / MODEL / ALT MODEL ÇEKME
    # ══════════════════════════════════════════════════════════════

    def get_brands(self, category=None):
        """Bir kategorideki tüm markaları çek"""
        if category is None:
            category = self.categories[0]

        cat_path = category['path']
        cat_slug = category['slug']
        cat_url  = f"{self.base_url}{cat_path}?take=50"

        print(f"📋 [{category['name']}] Markalar çekiliyor...")
        soup = self.get_page(cat_url, request_type='listing')

        brands = []
        if not soup:
            return brands

        items = soup.select(
            f'a[href*="{cat_path}/"][href$="?take=50"]'
        )
        seen = set()

        for item in items:
            href = item.get('href', '')
            path = href.replace('?take=50', '').split('/')[-1]

            if (self._is_brand_slug(path) and path
                    and path != cat_slug and href not in seen):
                seen.add(href)
                text = item.get_text(strip=True)
                name = re.sub(r'[\d\.\,]+', '', text).strip()
                if name:
                    brands.append({
                        'name': name,
                        'slug': path,
                        'url': urljoin(self.base_url, href),
                    })

        print(f"✅ {len(brands)} marka bulundu\n")
        return brands

    def get_models(self, brand_slug, brand_url, cat_path='/ikinci-el/otomobil'):
        """Marka altındaki modelleri çek"""
        soup = self.get_page(brand_url, request_type='listing')
        models = []
        if not soup:
            return models

        pattern = f'{cat_path}/{brand_slug}-'   # ← BURADA DEĞİŞTİ
        items = soup.select(f'a[href*="{pattern}"]')
        seen = set()

        # ... geri kalanı AYNI ...
        for item in items:
            href = item.get('href', '')
            if '?take=50' not in href:
                continue
            clean = href.split('?')[0]
            slug = clean.split('/')[-1]
            parts = slug.replace(f'{brand_slug}-', '').split('-')

            if (href not in seen and slug != brand_slug
                    and len(parts) <= 2):
                seen.add(href)
                ne = item.select_one('.list-name')
                ce = item.select_one('.count')
                name = ne.get_text(strip=True) if ne else re.sub(
                    r'[\d\.\,]+$', '', item.get_text(strip=True)
                ).strip()
                cs = ce.get_text(strip=True) if ce else "0"
                try:
                    count = int(cs.replace('.', '').replace(',', ''))
                except:
                    count = 0
                if name and count > 0:
                    models.append({
                        'name': name,
                        'slug': slug,
                        'url': urljoin(self.base_url, href),
                        'count': count,
                    })

        unique, seen_s = [], set()
        for m in models:
            if m['slug'] not in seen_s:
                seen_s.add(m['slug'])
                unique.append(m)
        return unique

    def get_sub_models(self, model_slug, model_url, cat_path='/ikinci-el/otomobil'):
        """Model altındaki alt modelleri çek"""
        soup = self.get_page(model_url, request_type='listing')
        subs = []
        if not soup:
            return subs

        pattern = f'{cat_path}/{model_slug}-'   # ← BURADA DEĞİŞTİ
        items = soup.select(f'a[href*="{pattern}"]')
        seen = set()

        # ... geri kalanı AYNI ...
        for item in items:
            href = item.get('href', '')
            if '?take=50' not in href:
                continue
            slug = href.split('?')[0].split('/')[-1]
            if href not in seen and slug != model_slug:
                seen.add(href)
                ne = item.select_one('.list-name')
                ce = item.select_one('.count')
                name = ne.get_text(strip=True) if ne else re.sub(
                    r'[\d\.\,]+$', '', item.get_text(strip=True)
                ).strip()
                cs = ce.get_text(strip=True) if ce else "0"
                try:
                    count = int(cs.replace('.', '').replace(',', ''))
                except:
                    count = 0
                if name and count > 0:
                    subs.append({
                        'name': name,
                        'slug': slug,
                        'url': urljoin(self.base_url, href),
                        'count': count,
                    })
        return subs

    # ══════════════════════════════════════════════════════════════
    #  KATEGORİ TARAMA (SAYFA SAYFA → ilan_no + fiyat topla)
    # ══════════════════════════════════════════════════════════════

    def scan_category_pages(self, base_url, category_name, max_pages=50):
        """
        Bir kategorinin TÜM liste sayfalarını tara.
        Her satırdan ilan_no + fiyat çıkar → bugun_gorulen'e ekle.
        DETAY SAYFASINA GİRMEZ.
        """
        # İlk sayfayı çek + toplam sayfa bul
        first_items, total_pages = self.scan_list_page_with_total(base_url)
        total_pages = min(total_pages, max_pages)

        if not first_items and total_pages <= 1:
            return 0

        # İlk sayfanın ilanlarını kaydet
        added = 0
        for item in first_items:
            ino = item['ilan_no']
            if ino not in self.bugun_gorulen:
                self.bugun_gorulen[ino] = {
                    'fiyat': item['fiyat'],
                    'url': item['url'],
                }
                added += 1
        self.stats['taranan_ilan'] += len(first_items)
        self.stats['taranan_sayfa'] += 1

        print(f"         Sf 1/{total_pages} → {len(first_items)} ilan", end="")

        # Kalan sayfalar
        for page in range(2, total_pages + 1):
            try:
                sep = '&' if '?' in base_url else '?'
                page_url = f"{base_url}{sep}page={page}"

                items = self.scan_list_page(page_url)

                page_added = 0
                for item in items:
                    ino = item['ilan_no']
                    if ino not in self.bugun_gorulen:
                        self.bugun_gorulen[ino] = {
                            'fiyat': item['fiyat'],
                            'url': item['url'],
                        }
                        page_added += 1
                added += page_added
                self.stats['taranan_ilan'] += len(items)
                self.stats['taranan_sayfa'] += 1

                # Her 10 sayfada bir progress göster
                if page % 10 == 0 or page == total_pages:
                    print(f"\n         Sf {page}/{total_pages} → "
                        f"toplam {len(self.bugun_gorulen):,} ilan", end="")

            except KeyboardInterrupt:
                print("\n    ⛔ Durduruldu!")
                self.save_state(phase='scan')
                raise
            except Exception as e:
                print(f"\n    ⚠️ Sayfa hatası: {e}")
                if not self.wait_for_connection():
                    break
                continue

        print(f" ✓ +{added}")
        return added

    def scan_category_with_year_filter(self, category_url, category_name,
                                        start_year_idx=0):
        """
        2500+ ilan varsa yıl filtresi uygulayarak tara.
        Tek yıl bile 2500'ü geçerse... o kadarını alır (site limiti).
        """
        first_items, total_pages = self.scan_list_page_with_total(category_url)

        # İlk sayfanın ilanlarını kaydet
        for item in first_items:
            ino = item['ilan_no']
            if ino not in self.bugun_gorulen:
                self.bugun_gorulen[ino] = {
                    'fiyat': item['fiyat'],
                    'url': item['url'],
                }
        self.stats['taranan_ilan'] += len(first_items)
        self.stats['taranan_sayfa'] += 1

        if total_pages >= 50:
            print(f"       ⚠️ 2500+ ilan → yıl filtresi uygulanıyor")

            year_ranges = [
                (1980, 2004), (2005, 2008), (2009, 2011),
                (2012, 2012), (2013, 2013), (2014, 2014),
                (2015, 2015), (2016, 2016), (2017, 2017),
                (2018, 2018), (2019, 2019), (2020, 2020),
                (2021, 2021), (2022, 2022), (2023, 2023),
                (2024, 2024), (2025, 2026),
            ]

            for yr_idx in range(start_year_idx, len(year_ranges)):
                ymin, ymax = year_ranges[yr_idx]
                sep = '&' if '?' in category_url else '?'
                furl = f"{category_url}{sep}minYear={ymin}&maxYear={ymax}"

                if ymin == ymax:
                    print(f"       📅 {ymin}:")
                else:
                    print(f"       📅 {ymin}-{ymax}:")

                # Yıl filtresi de 50 sayfayı aşıyorsa → fiyat kırılımı uygula
                yr_items, yr_pages = self.scan_list_page_with_total(furl)

                # İlk sayfanın ilanlarını kaydet
                for item in yr_items:
                    ino = item['ilan_no']
                    if ino not in self.bugun_gorulen:
                        self.bugun_gorulen[ino] = {
                            'fiyat': item['fiyat'],
                            'url': item['url'],
                        }
                self.stats['taranan_ilan'] += len(yr_items)
                self.stats['taranan_sayfa'] += 1

                if yr_pages >= 50:
                    print(f"         ⚠️ Yıl filtresi de yetmedi → fiyat kırılımı")
                    price_ranges = [
                        (0, 300000), (300000, 500000), (500000, 750000),
                        (750000, 1000000), (1000000, 1500000), (1500000, 99999999)
                    ]
                    for pmin, pmax in price_ranges:
                        pfurl = f"{furl}&minPrice={pmin}&maxPrice={pmax}"
                        print(f"         💰 {pmin:,}-{pmax:,} TL:")
                        self.scan_category_pages(
                            pfurl, 
                            f"{category_name} ({ymin}-{ymax}, {pmin}-{pmax}TL)"
                        )
                else:
                    # Yıl filtresi yeterli, kalan sayfaları normal tara
                    for page in range(2, yr_pages + 1):
                        sep2 = '&' if '?' in furl else '?'
                        page_url = f"{furl}{sep2}page={page}"
                        items = self.scan_list_page(page_url)
                        for item in items:
                            ino = item['ilan_no']
                            if ino not in self.bugun_gorulen:
                                self.bugun_gorulen[ino] = {
                                    'fiyat': item['fiyat'],
                                    'url': item['url'],
                                }
                        self.stats['taranan_ilan'] += len(items)
                        self.stats['taranan_sayfa'] += 1

                        if page % 10 == 0 or page == yr_pages:
                            print(f"         Sf {page}/{yr_pages} → "
                                f"{len(self.bugun_gorulen):,} toplam")

        else:
            # İlk sayfa zaten çekildi, kalan sayfaları tara
            if total_pages > 1:
                print(f"       {total_pages} sayfa taranacak")
                # 2. sayfadan devam et (1. sayfayı zaten çektik)
                for page in range(2, total_pages + 1):
                    try:
                        sep = '&' if '?' in category_url else '?'
                        page_url = f"{category_url}{sep}page={page}"

                        items = self.scan_list_page(page_url)
                        for item in items:
                            ino = item['ilan_no']
                            if ino not in self.bugun_gorulen:
                                self.bugun_gorulen[ino] = {
                                    'fiyat': item['fiyat'],
                                    'url': item['url'],
                                }
                        self.stats['taranan_ilan'] += len(items)
                        self.stats['taranan_sayfa'] += 1

                        if page % 10 == 0 or page == total_pages:
                            print(f"         Sf {page}/{total_pages} → "
                                f"{len(self.bugun_gorulen):,} toplam")

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"\n    ⚠️ Sayfa hatası: {e}")
                        if not self.wait_for_connection():
                            break
                        continue

    # ══════════════════════════════════════════════════════════════
    #  ⭐ AŞAMA 1: TÜM SİTEYİ TARA (SADECE ilan_no + fiyat)
    # ══════════════════════════════════════════════════════════════

    def asama1_tum_siteyi_tara(self, start_brand_idx=0,
                            start_model_idx=0, start_cat_idx=0):
        """
        Tüm arabam.com ilanlarını tara.
        ✅ Otomobil + Arazi, SUV, Pick-up
        
        YENİ STRATEJİ (Sızıntısız):
        1. Önce MARKA sayfasını direkt tara (yıl filtresi ile)
        → Modele atanmamış ilanlar yakalanır
        2. Sonra her modeli direkt tara (yıl filtresi ile)
        → Sub_model'e atanmamış ilanlar yakalanır
        → Marka taraması yetmediyse güvenlik ağı görevi görür
        
        Dict unique olduğu için duplicate sorun değil.
        """
        print("=" * 65)
        print("📋 AŞAMA 1: TÜM İLANLARI TARA (sadece ilan_no + fiyat)")
        print("   ✅ Otomobil + Arazi, SUV, Pick-up")
        print("   ✅ Sızıntısız mod: Marka + Model çift tarama")
        print("=" * 65)

        for cat_idx in range(start_cat_idx, len(self.categories)):
            category = self.categories[cat_idx]
            cat_path = category['path']

            print(f"\n{'═'*65}")
            print(f"🗂️  KATEGORİ: {category['name']}")
            print(f"{'═'*65}")

            brands = self.get_brands(category)
            if not brands:
                print(f"⚠️ {category['name']} için marka bulunamadı!")
                continue

            total_brands = len(brands)
            b_start = start_brand_idx if cat_idx == start_cat_idx else 0

            for brand_idx in range(b_start, total_brands):
                brand = brands[brand_idx]

                try:
                    print(f"\n{'─'*60}")
                    print(f"🏷️ [{brand_idx+1}/{total_brands}] "
                        f"{brand['name']} ({category['name']})")
                    print(f"   Şu ana kadar: {len(self.bugun_gorulen):,} ilan")
                    print(f"{'─'*60}")

                    # ═══════════════════════════════════════════════════
                    # ⭐ ADIM 1: MARKAYI DİREKT TARA (YENİ)
                    # Modele atanmamış ilanları yakalamak için kritik!
                    # ═══════════════════════════════════════════════════
                    onceki_sayi = len(self.bugun_gorulen)
                    print(f"   🌐 Marka sayfası direkt taranıyor...")
                    
                    try:
                        self.scan_category_with_year_filter(
                            brand['url'], 
                            f"{brand['name']} (marka geneli)"
                        )
                        marka_eklenen = len(self.bugun_gorulen) - onceki_sayi
                        print(f"   ✓ Marka taraması: +{marka_eklenen:,} ilan")
                    except Exception as e:
                        print(f"   ⚠️ Marka direkt tarama hatası: {e}")

                    # ═══════════════════════════════════════════════════
                    # ADIM 2: MODELLERİ TARA (Güvenlik ağı + detaylı kapsam)
                    # ═══════════════════════════════════════════════════
                    models = self.get_models(
                        brand['slug'], brand['url'], cat_path
                    )
                    print(f"   📁 {len(models)} model bulundu")

                    m_start = (start_model_idx
                            if (cat_idx == start_cat_idx
                                and brand_idx == b_start)
                            else 0)

                    for model_idx in range(m_start, len(models)):
                        model = models[model_idx]

                        self.save_state(
                            phase='scan',
                            cat_idx=cat_idx,
                            brand_idx=brand_idx,
                            model_idx=model_idx,
                        )

                        print(f"\n  🚙 [{model_idx+1}/{len(models)}] "
                            f"{model['name']} ({model['count']})")

                        try:
                            cat = f"{brand['name']} > {model['name']}"
                            
                            # ═══════════════════════════════════════════
                            # ⭐ DEĞİŞİKLİK: Sub_models'e GİRME!
                            # Sub_model'e atanmamış ilanlar kaçar.
                            # Yıl filtresi 2500+'ı zaten çözüyor.
                            # ═══════════════════════════════════════════
                            if model['count'] > 2500:
                                # Yıl filtresi devreye girsin
                                # (50 sayfa aşılırsa fiyat filtresi de var)
                                self.scan_category_with_year_filter(
                                    model['url'], cat
                                )
                            else:
                                # Normal sayfalama yeterli
                                self.scan_category_pages(model['url'], cat)

                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            print(f"    ⚠️ Model hatası: {e}")
                            continue

                    print(f"\n   ✅ {brand['name']} tamamlandı → "
                        f"toplam {len(self.bugun_gorulen):,} ilan")

                except KeyboardInterrupt:
                    print("\n\n⛔ DURDURULDU!")
                    self.save_state(phase='scan',
                                    cat_idx=cat_idx,
                                    brand_idx=brand_idx)
                    raise
                except Exception as e:
                    print(f"\n⚠️ Marka hatası: {e}")
                    self.save_state(phase='scan',
                                    cat_idx=cat_idx,
                                    brand_idx=brand_idx)
                    if not self.wait_for_connection():
                        return False
                    continue

            print(f"\n  🗂️ ✅ {category['name']} tamamlandı!")

        print(f"\n{'='*65}")
        print(f"✅ AŞAMA 1 TAMAMLANDI!")
        print(f"   Taranan sayfa : {self.stats['taranan_sayfa']:,}")
        print(f"   Bulunan ilan  : {len(self.bugun_gorulen):,}")
        print(f"{'='*65}")
        return True
    # ══════════════════════════════════════════════════════════════
    #  AŞAMA 1-B: DB İLE KARŞILAŞTIR → Yeni/Değişen/Satılan Bul
    # ══════════════════════════════════════════════════════════════

    def asama1b_karsilastir(self):
        """
        bugun_gorulen vs db_aktif karşılaştır:

        1. Sitede VAR + DB'de YOK     → yeni_ilan_urls listesine ekle
        2. Sitede VAR + DB'de VAR      → fiyat değiştiyse güncelle
                                        + son_gorulme güncelle
        3. DB'de VAR + Sitede YOK      → "satıldı" adayı
        """
        print(f"\n{'='*65}")
        print(f"🔍 AŞAMA 1-B: DB İLE KARŞILAŞTIRMA")
        print(f"   Sitede görülen : {len(self.bugun_gorulen):,}")
        print(f"   DB'de aktif    : {len(self.db_aktif):,}")
        print(f"{'='*65}\n")

        yeni_count = 0
        fiyat_degisen = 0
        son_gorulme = 0

        # ═══ 1. SİTEDE OLAN HER İLANI KONTROL ET ═══
        bugun_ilan_nolari = set(self.bugun_gorulen.keys())

        batch_updates = []  # (ilan_id, fiyat) → toplu UPDATE
        batch_gorulme = []  # (ilan_id,) → toplu son_gorulme UPDATE

        for ilan_no, info in self.bugun_gorulen.items():
            if ilan_no in self.db_aktif:
                # DB'de var → güncelle
                db_id, db_fiyat = self.db_aktif[ilan_no]
                site_fiyat = info.get('fiyat')

                # Fiyat değişmiş mi?
                if (site_fiyat and db_fiyat and
                        abs(site_fiyat - db_fiyat) > 1):
                    batch_updates.append((db_id, site_fiyat))
                    fiyat_degisen += 1
                else:
                    batch_gorulme.append((db_id,))
                    son_gorulme += 1
            else:
                # DB'de yok → yeni ilan
                url = info.get('url')
                if url:
                    self.yeni_ilan_urls.append({
                        'ilan_no': ilan_no,
                        'url': url,
                        'fiyat': info.get('fiyat'),
                    })
                    yeni_count += 1

        # ═══ TOPLU DB GÜNCELLEMELERİ ═══
        print(f"  📝 Fiyat değişen {fiyat_degisen:,} ilan güncelleniyor...")
        if batch_updates:
            try:
                for db_id, new_fiyat in batch_updates:
                    self.db_cursor.execute("""
                        UPDATE ilanlar SET
                            fiyat = %s,
                            son_gorulme_tarihi = NOW(),
                            updated_at = NOW()
                        WHERE id = %s
                    """, (new_fiyat, db_id))

                    # Fiyat geçmişine de kaydet
                    self.db_cursor.execute("""
                        INSERT INTO fiyat_gecmisi (ilan_id, fiyat)
                        VALUES (%s, %s)
                    """, (db_id, new_fiyat))

                self.db_conn.commit()
                self.stats['fiyat_degisen'] = fiyat_degisen
                print(f"  ✅ {fiyat_degisen:,} fiyat güncellendi + geçmişe kaydedildi")
            except Exception as e:
                self.db_conn.rollback()
                print(f"  ❌ Fiyat güncelleme hatası: {e}")

        # son_gorulme toplu güncelle
        print(f"  📝 {son_gorulme:,} ilanın son_gorulme güncelleniyor...")
        if batch_gorulme:
            try:
                # Toplu güncelleme: tüm ID'ler için tek sorgu
                ids = [x[0] for x in batch_gorulme]
                # 1000'lik batch'ler halinde
                for i in range(0, len(ids), 1000):
                    chunk = ids[i:i+1000]
                    placeholders = ','.join(['%s'] * len(chunk))
                    self.db_cursor.execute(f"""
                        UPDATE ilanlar SET
                            son_gorulme_tarihi = NOW()
                        WHERE id IN ({placeholders})
                    """, chunk)
                self.db_conn.commit()
                self.stats['son_gorulme_guncellenen'] = son_gorulme
                print(f"  ✅ {son_gorulme:,} son_gorulme güncellendi")
            except Exception as e:
                self.db_conn.rollback()
                print(f"  ❌ son_gorulme güncelleme hatası: {e}")

        # ═══ 2. SATILAN ADAYLARI (DB'de aktif ama sitede yok) ═══
        db_aktif_set = set(self.db_aktif.keys())
        satilan_adaylari = db_aktif_set - bugun_ilan_nolari

        print(f"\n  📊 KARŞILAŞTIRMA SONUCU:")
        print(f"     Yeni ilan (detay çekilecek) : {yeni_count:,}")
        print(f"     Fiyat değişen               : {fiyat_degisen:,}")
        print(f"     Son görülme güncellenen      : {son_gorulme:,}")
        print(f"     Satılan adayı               : {len(satilan_adaylari):,}")
        print(f"{'='*65}")

        return {
            'yeni': yeni_count,
            'fiyat_degisen': fiyat_degisen,
            'son_gorulme': son_gorulme,
            'satilan_aday': len(satilan_adaylari),
            'satilan_ilan_nolari': satilan_adaylari,
        } 
    
    #  PART 3/4: YENİ İLAN DETAY ÇEKME + DB'YE YAZMA
#
#  Aşama 1'de bulunan yeni ilanların detay sayfasına girer.
#  Hasar, tramer, tüm özellikler çeker → DB'ye INSERT eder.
#  Sadece DB'de OLMAYAN ilanlar için çalışır.
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
#  HASAR BİLGİSİ ÇEKME (window.damage JSON)
# ──────────────────────────────────────────────────────────────

    def parse_damage_from_script(self, soup, data):
        """
        window.damage JSON'ından 13 parçanın durumunu çek.
        Bulunamazsa HTML fallback dene.
        """
        # Varsayılan değerler
        for col_name in self.part_columns.values():
            data[col_name] = 'belirtilmemis'

        data['boyali_sayi'] = 0
        data['degismis_sayi'] = 0
        data['lokal_boyali_sayi'] = 0
        data['orijinal_sayi'] = 0
        data['belirtilmemis_sayi'] = 0
        data['tramer_tutari'] = None

        # ═══ YÖNTEM 1: window.damage JSON ═══
        damage_found = False

        for script in soup.select('script'):
            if script.string and 'window.damage' in script.string:
                match = re.search(
                    r'window\.damage\s*=\s*(.∗?);', script.string, re.DOTALL )
                
                if match:   
                    try:
                        damage_list = json.loads(match.group(1))
                        damage_found = True
                        boyali, degismis, lokal = [], [], []
                        orijinal, belirtilmemis = [], []

                        for part in damage_list:
                            name = part.get('Name', '')
                            value = str(part.get('Value', '0'))
                            vt = part.get('ValueText', '')

                            if vt == 'painted' or value == '3':
                                durum = 'boyali'
                                boyali.append(name)
                            elif vt == 'changed' or value == '4':
                                durum = 'degismis'
                                degismis.append(name)
                            elif vt == 'localpainted' or value == '2':
                                durum = 'lokal_boyali'
                                lokal.append(name)
                            elif vt == 'original' or value == '1':
                                durum = 'orijinal'
                                orijinal.append(name)
                            else:
                                durum = 'belirtilmemis'
                                belirtilmemis.append(name)

                            if name in self.part_columns:
                                data[self.part_columns[name]] = durum

                        data['boyali_sayi'] = len(boyali)
                        data['degismis_sayi'] = len(degismis)
                        data['lokal_boyali_sayi'] = len(lokal)
                        data['orijinal_sayi'] = len(orijinal)
                        data['belirtilmemis_sayi'] = len(belirtilmemis)

                    except Exception as e:
                        print(f"      ⚠️ damage parse hatası: {e}")
                break

        # ═══ YÖNTEM 2: HTML fallback ═══
        if not damage_found:
            self._parse_damage_html_fallback(soup, data)

        # ═══ TRAMER ═══
        tramer = soup.select_one('.tramer-info')
        if tramer:
            text = tramer.get_text(strip=True)
            if 'yok' in text.lower():
                data['tramer_tutari'] = 0
            else:
                m = re.search(r'([\d\.\,]+)', text)
                if m:
                    data['tramer_tutari'] = self.temizle_tramer(m.group(1))

    def _parse_damage_html_fallback(self, soup, data):
        """JSON bulunamazsa HTML'den hasar bilgisi çek"""
        section = soup.select_one('#tab-damage-information')
        if not section:
            section = soup.select_one('.damage-information-container')
        if not section:
            return

        for item in section.select('.car-damage-info-item'):
            title_elem = item.select_one('p')
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True).lower()
            parts = []
            for li in item.select('li'):
                pn = li.get_text(strip=True)
                if pn and pn != '-':
                    parts.append(pn)

            if 'orijinal' in title or 'orjinal' in title:
                durum = 'orijinal'
                data['orijinal_sayi'] = len(parts)
            elif 'lokal' in title:
                durum = 'lokal_boyali'
                data['lokal_boyali_sayi'] = len(parts)
            elif 'boyalı' in title or 'boyanmış' in title:
                durum = 'boyali'
                data['boyali_sayi'] = len(parts)
            elif 'değişmiş' in title or 'degismis' in title:
                durum = 'degismis'
                data['degismis_sayi'] = len(parts)
            elif 'belirtilmemiş' in title:
                durum = 'belirtilmemis'
                data['belirtilmemis_sayi'] = len(parts)
            else:
                continue

            for pn in parts:
                if pn in self.part_columns:
                    data[self.part_columns[pn]] = durum

    # ──────────────────────────────────────────────────────────────
    #  DETAY SAYFASINDAN TÜM VERİLERİ ÇEK
    # ──────────────────────────────────────────────────────────────

    def get_listing_detail(self, url):
        """
        İlan detay sayfasından TÜM verileri çek:
        ✅ İlan numarası
        ✅ Fiyat, km, yıl, marka, seri, model...
        ✅ 13 parça hasar durumu
        ✅ Tramer bilgisi
        """
        soup = self.get_page(url, request_type='detail')
        if not soup:
            self.failed_urls.append(url)
            return None

        data = {}

        # ═══ İLAN NO ═══
        ilan_no = self.extract_ilan_no_from_url(url)
        page_no = self.extract_ilan_no_from_page(soup)
        data['ilan_no'] = page_no or ilan_no or '-'
        data['ilan_url'] = url

        # ═══ FİYAT ═══
        price = soup.select_one('.desktop-information-price')
        if price:
            data['fiyat'] = price.get_text(strip=True)

                # ═══ KONUM ═══
        data['konum'] = None
        # Senin bulduğun güncel class: .product-location
        for sel in ['.product-location', '.classified-location']:
            konum_elem = soup.select_one(sel)
            if konum_elem:
                # Ekranda "Merkez Merkez, Adıyaman" yazan yeri tertemiz alır
                data['konum'] = konum_elem.get_text(strip=True)
                break
            
        # ═══ TÜM ÖZELLİKLER ═══
        field_mapping = {
            'ilan no': 'ilan_no_sayfa',
            'marka': 'marka',
            'seri': 'seri',
            'model': 'model',
            'yil': 'yil', 'yıl': 'yil',
            'kilometre': 'kilometre',
            'vites tipi': 'vites_tipi',
            'yakit tipi': 'yakit_tipi', 'yakıt tipi': 'yakit_tipi',
            'kasa tipi': 'kasa_tipi',
            'renk': 'renk',
            'motor hacmi': 'motor_hacmi',
            'motor gucu': 'motor_gucu', 'motor gücü': 'motor_gucu',
            'kimden': 'kimden',
            'ilan tarihi': 'ilan_tarihi',
            'cekis': 'cekis', 'çekiş': 'cekis',
        }

        for item in soup.select('.property-item'):
            ke = item.select_one('.property-key')
            ve = item.select_one('.property-value')
            if ke and ve:
                key = ke.get_text(strip=True).lower().strip()
                val = ve.get_text(strip=True)
                key = (key.replace('i̇', 'i')
                    .replace('ı', 'i')
                    .replace('İ', 'i'))
                if key in field_mapping:
                    data[field_mapping[key]] = val

        # İlan no düzeltme
        if data.get('ilan_no_sayfa') and data['ilan_no'] == '-':
            data['ilan_no'] = data.pop('ilan_no_sayfa')
        else:
            data.pop('ilan_no_sayfa', None)

        # ═══ İLAN AÇIKLAMASI (Gemini analizi için) ═══
        data['ilan_aciklama'] = None
        desc_section = soup.select_one('#tab-description')
        if desc_section:
            # h5 başlığını atla, içerik div'ini al
            desc_div = desc_section.find('div')
            if desc_div:
                # <br> taglerini newline'a çevir (paragraflar korunsun)
                for br in desc_div.find_all('br'):
                    br.replace_with('\n')
                text = desc_div.get_text(separator='\n', strip=True)
                # Çoklu boş satırları tek satıra indir
                text = re.sub(r'\n\s*\n+', '\n', text).strip()
                if text:
                    data['ilan_aciklama'] = text

    

        # ═══ HASAR + TRAMER ═══
        self.parse_damage_from_script(soup, data)

        return data

    # ──────────────────────────────────────────────────────────────
    #  DB'YE YENİ İLAN EKLEME
    # ──────────────────────────────────────────────────────────────

    def db_insert_new_ilan(self, data):
        """
        Tek bir yeni ilanı veritabanına INSERT et.
        Fiyat geçmişine de ilk kaydı ekle.
        Return: 'yeni' | 'hatali' | 'var_zaten'
        """
        try:
            ilan_no = str(data.get('ilan_no', '')).strip()
            if not ilan_no or ilan_no == '-':
                return 'hatali'

            fiyat = self.parse_fiyat(data.get('fiyat'))
            km = self.parse_km(data.get('kilometre'))
            yil = self.parse_yil(data.get('yil'))
            tarih = self.parse_tarih(data.get('ilan_tarihi'))
            tramer = data.get('tramer_tutari')
            if isinstance(tramer, str):
                tramer = self.temizle_tramer(tramer)

            # DB'de zaten var mı? (double check)
            self.db_cursor.execute(
                "SELECT id FROM ilanlar "
                "WHERE kaynak = 'arabam' AND kaynak_ilan_no = %s",
                (ilan_no,)
            )
            if self.db_cursor.fetchone():
                return 'var_zaten'

            # JSONB ekstra veriler
            ekstra = {}
            for col in ['orijinal_parcalar', 'boyali_parcalar',
                        'lokal_boyali_parcalar', 'degismis_parcalar']:
                val = data.get(col)
                if val and str(val).strip() not in ['-', '', 'nan']:
                    ekstra[col] = str(val).strip()

            # ═══ INSERT ═══
            self.db_cursor.execute("""
                INSERT INTO ilanlar (
                    kaynak, kaynak_ilan_no, ilan_url,
                    marka, seri, model, yil, kilometre, fiyat,
                    vites_tipi, yakit_tipi, kasa_tipi, renk,
                    motor_hacmi, motor_gucu, cekis,
                    kimden, ilan_tarihi, konum,
                    kaput, tavan, on_tampon, arka_tampon,
                    sol_on_camurluk, sag_on_camurluk,
                    sol_on_kapi, sag_on_kapi,
                    sol_arka_kapi, sag_arka_kapi,
                    sol_arka_camurluk, sag_arka_camurluk,
                    bagaj_kapagi,
                    tramer_tutari,
                    orijinal_sayi, boyali_sayi, lokal_boyali_sayi,
                    degismis_sayi, belirtilmemis_sayi,
                   ekstra_veriler,
                    ilan_aciklama,
                    ilan_durumu, ilk_gorulme_tarihi, son_gorulme_tarihi
                ) VALUES (
                    'arabam', %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s,
                    %s,
                    %s, %s, %s, %s, %s,
                    %s,
                    %s,
                    'aktif', NOW(), NOW()
                ) RETURNING id
            """, (
                ilan_no,
                data.get('ilan_url'),
                # Araç bilgileri
                data.get('marka'),
                data.get('seri'),
                data.get('model'),
                yil, km, fiyat,
                data.get('vites_tipi'),
                data.get('yakit_tipi'),
                data.get('kasa_tipi'),
                data.get('renk'),
                data.get('motor_hacmi'),
                data.get('motor_gucu'),
                data.get('cekis'),
                data.get('kimden'),
                tarih,
                data.get('konum'),
                # 13 parça (motor_kaputu → kaput, arka_kaput → bagaj_kapagi)
                self.temizle_parca(data.get('motor_kaputu')),
                self.temizle_parca(data.get('tavan')),
                self.temizle_parca(data.get('on_tampon')),
                self.temizle_parca(data.get('arka_tampon')),
                self.temizle_parca(data.get('sol_on_camurluk')),
                self.temizle_parca(data.get('sag_on_camurluk')),
                self.temizle_parca(data.get('sol_on_kapi')),
                self.temizle_parca(data.get('sag_on_kapi')),
                self.temizle_parca(data.get('sol_arka_kapi')),
                self.temizle_parca(data.get('sag_arka_kapi')),
                self.temizle_parca(data.get('sol_arka_camurluk')),
                self.temizle_parca(data.get('sag_arka_camurluk')),
                self.temizle_parca(data.get('arka_kaput')),
                # Sayılar
                tramer,
                data.get('orijinal_sayi', 0),
                data.get('boyali_sayi', 0),
                data.get('lokal_boyali_sayi', 0),
                data.get('degismis_sayi', 0),
                data.get('belirtilmemis_sayi', 0),
               # JSONB
                json.dumps(ekstra, ensure_ascii=False),
                # İlan açıklama metni (Gemini analizi için)
                data.get('ilan_aciklama'),
            ))

            new_id = self.db_cursor.fetchone()[0]

            # ═══ İlk fiyatı fiyat_gecmisi'ne kaydet ═══
            if fiyat:
                self.db_cursor.execute(
                    "INSERT INTO fiyat_gecmisi (ilan_id, fiyat) "
                    "VALUES (%s, %s)",
                    (new_id, fiyat)
                )

            self.db_conn.commit()
            return 'yeni'

        except Exception as e:
            self.db_conn.rollback()
            print(f"      ❌ DB insert hatası: {str(e)[:120]}")
            return 'hatali'

    # ──────────────────────────────────────────────────────────────
    #  SATILAN TESPİTİ + satilan_araclar TABLOSUNA YAZMA
    # ──────────────────────────────────────────────────────────────

    def asama_satilan_tespit(self, satilan_ilan_nolari):
        """
        Sitede görünmeyen aktif ilanları "satıldı" olarak işaretle.

        Mantık:
        - Bugün sitede görünMEYEN + DB'de aktif olan ilanlar
        - İlk gün: 2+ gün üst üste görünmezse satıldı say
            (tek güne güvenme, site geçici olarak göstermeyebilir)
        - Burada: son_gorulme < bugün - 1 gün olan ilanları satıldı say

        Bu fonksiyon:
        1. ilanlar.ilan_durumu → 'satildi'
        2. ilanlar.satilma_tarihi → NOW()
        3. satilan_araclar tablosuna kayıt ekle
        """
        if not satilan_ilan_nolari:
            print("  ✅ Satılan aday yok")
            return 0

        print(f"\n{'='*65}")
        print(f"🏷️ SATILAN TESPİTİ")
        print(f"   {len(satilan_ilan_nolari):,} aday kontrol ediliyor")
        print(f"{'='*65}")

        satilan_count = 0
        batch_size = 500

        satilan_list = list(satilan_ilan_nolari)

        for i in range(0, len(satilan_list), batch_size):
            chunk = satilan_list[i:i + batch_size]
            placeholders = ','.join(['%s'] * len(chunk))

            try:
                # ═══ 1. Son görülme 1+ gün önceyse satıldı say ═══
                # (Bugün ilk kez görünmeyenler → hemen satıldı demiyoruz,
                #  ama Aşama 1'de tüm siteyi taradığımız için burada
                #  gerçekten sitede olmayan ilanlar)
                self.db_cursor.execute(f"""
                    SELECT id, kaynak_ilan_no, kaynak, marka, seri, model,
                        yil, kilometre, fiyat, konum, ilan_tarihi,
                        ilk_gorulme_tarihi, son_gorulme_tarihi
                    FROM ilanlar
                    WHERE kaynak = 'arabam'
                    AND ilan_durumu = 'aktif'
                    AND kaynak_ilan_no IN ({placeholders})
                    AND son_gorulme_tarihi < NOW() - INTERVAL '3 days'
                """, chunk)

                rows = self.db_cursor.fetchall()

                for row in rows:
                    (db_id, k_ilan_no, kaynak, marka, seri, model,
                    yil, km, fiyat, konum, ilan_tarihi,
                    ilk_gorulme, son_gorulme) = row

                    # İlan süresini hesapla
                    ilan_suresi = None
                    if ilk_gorulme:
                        delta = datetime.now() - ilk_gorulme
                        ilan_suresi = delta.days

                    # İlk fiyatı bul
                    ilk_fiyat = None
                    try:
                        self.db_cursor.execute("""
                            SELECT fiyat FROM fiyat_gecmisi
                            WHERE ilan_id = %s
                            ORDER BY tarih ASC LIMIT 1
                        """, (db_id,))
                        ilk_fiyat_row = self.db_cursor.fetchone()
                        if ilk_fiyat_row:
                            ilk_fiyat = ilk_fiyat_row[0]
                    except:
                        pass

                    # Fiyat değişim sayısı
                    fiyat_degisim = 0
                    try:
                        self.db_cursor.execute("""
                            SELECT COUNT(*) FROM fiyat_gecmisi
                            WHERE ilan_id = %s
                        """, (db_id,))
                        cnt = self.db_cursor.fetchone()
                        if cnt:
                            fiyat_degisim = max(0, cnt[0] - 1)
                    except:
                        pass

                    # Fiyat değişim oranı
                                        # Fiyat değişim oranı
                    fiyat_oran = None
                    if ilk_fiyat and fiyat and float(ilk_fiyat) > 0:
                        fiyat_oran = round(
                            ((float(fiyat) - float(ilk_fiyat))
                            / float(ilk_fiyat)) * 100, 2
                        )
                        # --- ANOMALİ (KİRLİ VERİ) FİLTRESİ ---
                        # Galerici 10 TL girip sonra 1 Milyon TL yaparsa oran %9999 olur ve DB patlar.
                        # DB limitine (NUMERIC 5,2) uyması için sınırlandırıyoruz:
                        if fiyat_oran > 999.99:
                            fiyat_oran = 999.99
                        elif fiyat_oran < -99.99:
                            fiyat_oran = -99.99
                    # ═══ ilanlar tablosunu güncelle ═══
                    self.db_cursor.execute("""
                        UPDATE ilanlar SET
                            ilan_durumu = 'satildi',
                            satilma_tarihi = NOW(),
                            updated_at = NOW()
                        WHERE id = %s
                    """, (db_id,))

                    # ═══ satilan_araclar tablosuna ekle ═══
                    self.db_cursor.execute("""
                        INSERT INTO satilan_araclar (
                            ilan_id, kaynak, ilan_suresi_gun,
                            ilk_fiyat, son_fiyat,
                            fiyat_degisim_sayisi, fiyat_degisim_orani,
                            marka, seri, model, yil, kilometre, konum,
                            ilan_tarihi, ilk_gorulme, son_gorulme,
                            tahmini_satis_tarihi
                        ) VALUES (
                            %s, %s, %s,
                            %s, %s,
                            %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s,
                            NOW()
                        )
                    """, (
                        db_id, 'arabam', ilan_suresi,
                        ilk_fiyat, fiyat,
                        fiyat_degisim, fiyat_oran,
                        marka, seri, model, yil, km, konum,
                        ilan_tarihi, ilk_gorulme, son_gorulme,
                    ))

                    satilan_count += 1

                self.db_conn.commit()

                if (i + batch_size) % 2000 == 0:
                    print(f"     İşlenen: {i + len(chunk):,} / "
                        f"{len(satilan_list):,} → "
                        f"{satilan_count:,} satıldı")

            except Exception as e:
                self.db_conn.rollback()
                print(f"  ❌ Satılan tespit hatası: {e}")
                continue

        self.stats['satilan'] = satilan_count
        print(f"\n  ✅ {satilan_count:,} ilan 'satıldı' olarak işaretlendi")
        print(f"     + satilan_araclar tablosuna kaydedildi")

        return satilan_count

    # ──────────────────────────────────────────────────────────────
    #  ⭐ AŞAMA 2: YENİ İLANLARIN DETAYINI ÇEK + DB'YE YAZ
    # ──────────────────────────────────────────────────────────────

    def asama2_yeni_ilanlari_cek(self):
        """
        Aşama 1'de bulunan yeni ilanların detay sayfasına gir.
        Hasar, tramer, tüm özellikleri çek → DB'ye INSERT et.
        """
        toplam = len(self.yeni_ilan_urls)

        if toplam == 0:
            print("\n✅ Yeni ilan yok, tüm ilanlar zaten DB'de!")
            return 0

        print(f"\n{'='*65}")
        print(f"📥 AŞAMA 2: YENİ İLANLARIN DETAYI ÇEKİLİYOR")
        print(f"   Toplam: {toplam:,} yeni ilan")
        print(f"   Tahmini süre: ~{toplam * 1.5 / 60:.0f} dakika")
        print(f"{'='*65}\n")

        basarili = 0
        hatali = 0

        for idx, ilan_info in enumerate(self.yeni_ilan_urls):
            try:
                url = ilan_info['url']
                ilan_no = ilan_info['ilan_no']

                # Progress (her 50'de bir)
                if (idx + 1) % 50 == 0 or idx == 0:
                    pct = (idx + 1) / toplam * 100
                    print(f"  [{idx+1:,}/{toplam:,}] ({pct:.1f}%) "
                        f"✅ {basarili:,} ❌ {hatali:,}")

                # Detay çek
                detail = self.get_listing_detail(url)

                if not detail:
                    hatali += 1
                    self.stats['hatali'] += 1
                    continue

                # DB'ye yaz
                result = self.db_insert_new_ilan(detail)

                if result == 'yeni':
                    basarili += 1
                    self.stats['yeni_ilan'] += 1
                elif result == 'var_zaten':
                    pass  # Arada başkası eklemiş olabilir
                else:
                    hatali += 1
                    self.stats['hatali'] += 1

                # Her 200 ilanda durum kaydet
                if (idx + 1) % 200 == 0:
                    self.save_state(phase='detail')

            except KeyboardInterrupt:
                print(f"\n\n⛔ DURDURULDU! "
                    f"{basarili:,} eklendi, {hatali:,} hatalı")
                self.save_state(phase='detail')
                raise
            except Exception as e:
                print(f"    ⚠️ İlan hatası ({ilan_no}): {e}")
                hatali += 1
                self.stats['hatali'] += 1
                continue

        print(f"\n{'='*65}")
        print(f"✅ AŞAMA 2 TAMAMLANDI!")
        print(f"   Başarılı : {basarili:,}")
        print(f"   Hatalı   : {hatali:,}")
        print(f"   Toplam   : {toplam:,}")
        print(f"{'='*65}")

        return basarili 
    
        # ══════════════════════════════════════════════════════════════
#  PART 4/4: ANA ÇALIŞTIRMA + ORKESTRASYON + RAPORLAMA
#
#  Tüm aşamaları sırasıyla çalıştırır:
#    Aşama 1   → Tüm siteyi tara (ilan_no + fiyat)
#    Aşama 1-B → DB ile karşılaştır
#    Aşama 2   → Yeni ilanların detayını çek
#    Aşama 3   → Satılanları tespit et
#    Rapor     → Detaylı bitiş raporu + scrape_log
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
#  ⭐ ANA FONKSİYON: GÜNLÜK STOK TAKİP
# ──────────────────────────────────────────────────────────────

    def run(self):
        """
        GÜNLÜK STOK TAKİP - Ana Orkestrasyon

        Her gün 1 kere çalıştır:
        1. Tüm siteyi tara → ilan_no + fiyat topla (detaya GİRMEZ)
        2. DB ile karşılaştır → yeni/değişen/satılan bul
        3. Yeni ilanların detayına gir → DB'ye ekle
        4. Satılanları işaretle → satilan_araclar'a yaz
        5. Rapor + log
        """
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  🚗 ARABAM.COM GÜNLÜK STOK TAKİP SİSTEMİ              ║")
        print("║                                                        ║")
        print("║  Aşama 1: Tüm siteyi tara (sadece ilan_no + fiyat)    ║")
        print("║  Aşama 2: DB ile karşılaştır → yeni/değişen/satılan   ║")
        print("║  Aşama 3: Yeni ilanların detayını çek → DB'ye ekle    ║")
        print("║  Aşama 4: Satılanları tespit et + kaydet               ║")
        print("║                                                        ║")
        print("║  ✅ 2500+ ilana dayanıklı (yıl filtresi)              ║")
        print("║  ✅ İnternet kesintilerine dayanıklı                   ║")
        print("║  ✅ Anti-ban (UA rotasyonu, akıllı gecikme)            ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"\n🕐 Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.stats['baslangic'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # ═══ DB BAĞLANTISI ═══
        if not self.db_connect():
            print("❌ DB bağlantısı kurulamadı! Çıkılıyor.")
            return False

        # Aktif ilanları yükle
        self.db_load_aktif_ilanlar()

        # Scrape log başlat
        self.db_log_start()

                # ═══ KALDIGI YERDEN DEVAM KONTROLÜ ═══
        start_brand_idx = 0
        start_model_idx = 0
        start_cat_idx = 0          # ← YENİ
        resume_phase = None

        saved = self.load_state()
        if saved:
            choice = input("\n🔄 Bugünkü kaldığı yerden devam? (E/h): ").strip().lower()
            if choice != 'h':
                resume_phase = saved.get('phase', '')
                start_brand_idx = saved.get('brand_idx', 0)
                start_model_idx = saved.get('model_idx', 0)
                start_cat_idx = saved.get('cat_idx', 0)        # ← bu da eksikti

                # HER ZAMAN geri yükle
                self.bugun_gorulen = saved.get('bugun_gorulen', {})
                self.yeni_ilan_urls = saved.get('yeni_ilan_urls', [])
                if saved.get('stats'):
                    self.stats.update(saved['stats'])

                # ERROR özel durumu
                if resume_phase == 'error':
                    resume_phase = 'satilan'
                    print("  ⚠️ Önceki çalışma hatayla bitmiş → Satılan tespitinden devam")

                print(f"⏩ '{resume_phase}' aşamasından devam ediliyor")
                print(f"   Geri yüklenen: {len(self.bugun_gorulen):,} ilan")
            else:
                print("🔄 Sıfırdan başlanıyor\n")

        success = True

        try:
            # ══════════════════════════════════════════════════════
            #  AŞAMA 1: TÜM SİTEYİ TARA
            # ══════════════════════════════════════════════════════
            if not resume_phase or resume_phase == 'scan':
                print("\n" + "▓" * 65)
                print("▓  AŞAMA 1/4: TÜM SİTEYİ TARA (ilan_no + fiyat)")
                print("▓" * 65)

                ok = self.asama1_tum_siteyi_tara(
                    start_brand_idx=start_brand_idx,
                    start_model_idx=start_model_idx,
                    start_cat_idx=start_cat_idx,       # ← YENİ
                )

                if not ok:
                    print("❌ Aşama 1 başarısız!")
                    success = False

                # Aşama 1 bitti → state kaydet
                self.save_state(phase='compare')
                start_brand_idx = 0
                start_model_idx = 0

                        # ══════════════════════════════════════════════════════
            #  AŞAMA 1-B: DB İLE KARŞILAŞTIR
            # ══════════════════════════════════════════════════════
            if success and (not resume_phase or resume_phase in ['scan', 'compare']):
                print("\n" + "▓" * 65)
                print("▓  AŞAMA 2/4: DB İLE KARŞILAŞTIR")
                print("▓" * 65)
                
                self.check_db_connection() # <--- BURAYA EKLEDİK (Koptuysa bağlanacak)
                
                # UYARI: Bir önceki mesajımda dediğim self.karsilastirma düzeltmesini unutma!
                self.karsilastirma = self.asama1b_karsilastir()

                self.save_state(phase='detail')

            # ══════════════════════════════════════════════════════
            #  AŞAMA 2: YENİ İLANLARIN DETAYINI ÇEK
            # ══════════════════════════════════════════════════════
            if success and (not resume_phase or resume_phase in ['scan', 'compare', 'detail']):
                print("\n" + "▓" * 65)
                print("▓  AŞAMA 3/4: YENİ İLANLARIN DETAYINI ÇEK")
                print("▓" * 65)
                
                self.check_db_connection() # <--- BURAYA DA EKLEYELİM

                self.asama2_yeni_ilanlari_cek()

                self.save_state(phase='satilan')
            # ══════════════════════════════════════════════════════
            #  AŞAMA 3: SATILANLARI TESPİT ET
            # ══════════════════════════════════════════════════════
            if success and (not resume_phase or resume_phase in
                        ['scan', 'compare', 'detail', 'satilan']):
                print("\n" + "▓" * 65)
                print("▓  AŞAMA 4/4: SATILANLARI TESPİT ET")
                print("▓" * 65)

                self.check_db_connection()

                satilan_nolar = set(self.db_aktif.keys()) - set(self.bugun_gorulen.keys())

                self.asama_satilan_tespit(satilan_nolar)

        except KeyboardInterrupt:
            print("\n\n╔══════════════════════════════════════════╗")
            print("║  ⛔ KULLANICI TARAFINDAN DURDURULDU!     ║")
            print("║  💾 Durum kaydedildi, yarın devam edin   ║")
            print("╚══════════════════════════════════════════╝")
            success = False

        except Exception as e:
            print(f"\n❌ KRİTİK HATA: {e}")
            self.save_state(phase='error')
            success = False

        finally:
            # ═══ FİNAL RAPOR ═══
            self._print_final_report()

            # ═══ DB LOG GÜNCELLE ═══
            durum = 'basarili' if success else 'yarim_kaldi'
            self.db_log_end(durum)

            # ═══ STATE TEMİZLE (başarılıysa) ═══
            if success:
                self.clear_state()

            # ═══ BAŞARISIZ URL'LERİ KAYDET ═══
            if self.failed_urls:
                try:
                    ts = datetime.now().strftime('%Y%m%d')
                    fname = f'failed_urls_{ts}.txt'
                    with open(fname, 'w') as f:
                        f.write('\n'.join(set(self.failed_urls)))
                    print(f"⚠️ {len(set(self.failed_urls))} başarısız URL → {fname}")
                except:
                    pass

            # ═══ DB BAĞLANTISINI KAPAT ═══
            self.db_disconnect()

        return success
    

    # ──────────────────────────────────────────────────────────────
    #  SADECE YENİ İLANLARI ÇEK (HIZLI MOD - days=1 filtresi)
    # ──────────────────────────────────────────────────────────────

    def run_sadece_yeniler(self):
        """HIZLI MOD: Son 24 saat (Otomobil + SUV)"""
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  ⚡ HIZLI MOD: YENİ İLANLAR (Otomobil + SUV)          ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.stats['baslangic'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if not self.db_connect():
            return False

        self.db_load_aktif_ilanlar()
        self.db_log_start()

        try:
            # ═══ HER KATEGORİ İÇİN ═══
            for category in self.categories:
                print(f"\n{'─'*60}")
                print(f"🗂️  {category['name']} - Son 24 saat")
                print(f"{'─'*60}")

                cat_url = f"{self.base_url}{category['path']}?days=1&take=50"

                first_items, total_pages = self.scan_list_page_with_total(cat_url)
                print(f"📊 {total_pages} sayfa")

                if total_pages >= 50:
                    print("⚠️ 2500+! Marka bazlı taranacak...")
                    self._yeniler_marka_bazli(category)
                else:
                    # İlk sayfa
                    for item in first_items:
                        ino = item['ilan_no']
                        if ino not in self.bugun_gorulen:
                            self.bugun_gorulen[ino] = {
                                'fiyat': item['fiyat'],
                                'url': item['url'],
                            }
                    self.stats['taranan_sayfa'] += 1
                    self.stats['taranan_ilan'] += len(first_items)

                    # Kalan sayfalar
                    for page in range(2, total_pages + 1):
                        page_url = f"{cat_url}&page={page}"
                        items = self.scan_list_page(page_url)
                        for item in items:
                            ino = item['ilan_no']
                            if ino not in self.bugun_gorulen:
                                self.bugun_gorulen[ino] = {
                                    'fiyat': item['fiyat'],
                                    'url': item['url'],
                                }
                        self.stats['taranan_sayfa'] += 1
                        self.stats['taranan_ilan'] += len(items)

                        if page % 10 == 0:
                            print(f"  Sf {page}/{total_pages} → "
                                f"{len(self.bugun_gorulen):,}")

            # ═══ YENİ İLANLARI BUL ═══
            print(f"\n📊 Toplam: {len(self.bugun_gorulen):,}")
            for ilan_no, info in self.bugun_gorulen.items():
                if ilan_no not in self.db_aktif:
                    self.yeni_ilan_urls.append({
                        'ilan_no': ilan_no,
                        'url': info['url'],
                        'fiyat': info['fiyat'],
                    })
            print(f"🆕 Yeni: {len(self.yeni_ilan_urls):,}")

            if self.yeni_ilan_urls:
                self.asama2_yeni_ilanlari_cek()

        except KeyboardInterrupt:
            print("\n⛔ Durduruldu!")
        finally:
            self._print_final_report()
            self.db_log_end('basarili')
            self.db_disconnect()

        return True
        # ──────────────────────────────────────────────────────────────
    #  MOD 3: SATILANLARI KURTARMA (Hata Sonrası Telafi)
    # ──────────────────────────────────────────────────────────────
        # ──────────────────────────────────────────────────────────────
    #  MOD 3: KESİN MATEMATİK İLE SATILANLARI KURTARMA
    # ──────────────────────────────────────────────────────────────
    def run_satilan_kurtarma(self):
        """
        Tarihlere güvenmek yerine, bugün taranan JSON dosyası ile 
        veritabanındaki aktif araçları doğrudan matematiksel (Set) olarak çıkarır.
        """
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  🛠️ MOD 3: JSON İLE KESİN SATILAN TESPİTİ              ║")
        print("╚══════════════════════════════════════════════════════════╝")
        
        if not self.db_connect():
            return False

        try:
            # 1. State dosyasından BUGÜN taranan 146 bin aracı yükle
            saved = self.load_state()
            if saved and 'bugun_gorulen' in saved:
                self.bugun_gorulen = saved['bugun_gorulen']
                print(f"\n📂 State dosyasından bugün taranan {len(self.bugun_gorulen):,} araç yüklendi.")
            else:
                print("\n❌ State dosyası bulunamadı! Bu işlem yapılamaz.")
                return

            # 2. Veritabanındaki tüm aktif ilanları yükle
            self.db_load_aktif_ilanlar()
            
            # 3. KUSURSUZ MATEMATİK: DB'de olup da, bugün sitede (JSON'da) olmayanlar
            satilan_adaylari = set(self.db_aktif.keys()) - set(self.bugun_gorulen.keys())
            
            print(f"\n📊 Veritabanında Aktif Görünen : {len(self.db_aktif):,}")
            print(f"📊 Bugün Sitede Görülen        : {len(self.bugun_gorulen):,}")
            print(f"⚠️ Aradaki Fark (Satılanlar)   : {len(satilan_adaylari):,}")
            
            if len(satilan_adaylari) == 0:
                print("\n✅ Kurtarılacak araç yok. Her şey güncel!")
            elif len(satilan_adaylari) > 15000:
                print("\n❌ DİKKAT: Fark çok büyük! Güvenlik sebebiyle işlem iptal edildi.")
            else:
                cevap = input("\nBu araçlar 'satıldı' olarak işlensin mi? (E/h): ")
                if cevap.lower() == 'e':
                    self.asama_satilan_tespit(satilan_adaylari)
                    print("✅ Kurtarma işlemi başarıyla tamamlandı!")

        except Exception as e:
            print(f"❌ Hata: {e}")
        
        finally:
            self.db_disconnect()

    def _yeniler_marka_bazli(self, category=None):
        """days=1 2500+ ise marka bazlı tara"""
        if category is None:
            category = self.categories[0]

        brands = self.get_brands(category)
        if not brands:
            return

        cat_path = category['path']

        for bi, brand in enumerate(brands):
            try:
                brand_url = brand['url']
                sep = '&' if '?' in brand_url else '?'
                filtered_url = f"{brand_url}{sep}days=1"

                first_items, total_pages = self.scan_list_page_with_total(
                    filtered_url
                )
                if not first_items and total_pages <= 1:
                    continue

                for item in first_items:
                    ino = item['ilan_no']
                    if ino not in self.bugun_gorulen:
                        self.bugun_gorulen[ino] = {
                            'fiyat': item['fiyat'],
                            'url': item['url'],
                        }
                self.stats['taranan_sayfa'] += 1
                self.stats['taranan_ilan'] += len(first_items)

                print(f"  🏷️ [{bi+1}/{len(brands)}] {brand['name']} "
                    f"({category['name']}) → {total_pages} sf", end="")

                if total_pages >= 50:
                    print(f" ⚠️ 2500+, model bazlı")
                    models = self.get_models(
                        brand['slug'], brand['url'], cat_path
                    )
                    for model in models:
                        model_url = model['url']
                        sep2 = '&' if '?' in model_url else '?'
                        m_filtered = f"{model_url}{sep2}days=1"

                        m_items, m_pages = self.scan_list_page_with_total(
                            m_filtered
                        )
                        for item in m_items:
                            ino = item['ilan_no']
                            if ino not in self.bugun_gorulen:
                                self.bugun_gorulen[ino] = {
                                    'fiyat': item['fiyat'],
                                    'url': item['url'],
                                }
                        self.stats['taranan_sayfa'] += 1
                        self.stats['taranan_ilan'] += len(m_items)

                        if m_pages >= 50:
                            cat = f"{brand['name']} > {model['name']}"
                            self.scan_category_with_year_filter(
                                m_filtered, cat
                            )
                        else:
                            for pg in range(2, m_pages + 1):
                                pg_url = f"{m_filtered}&page={pg}"
                                pg_items = self.scan_list_page(pg_url)
                                for item in pg_items:
                                    ino = item['ilan_no']
                                    if ino not in self.bugun_gorulen:
                                        self.bugun_gorulen[ino] = {
                                            'fiyat': item['fiyat'],
                                            'url': item['url'],
                                        }
                                self.stats['taranan_sayfa'] += 1
                                self.stats['taranan_ilan'] += len(pg_items)
                else:
                    for pg in range(2, total_pages + 1):
                        pg_url = f"{filtered_url}&page={pg}"
                        pg_items = self.scan_list_page(pg_url)
                        for item in pg_items:
                            ino = item['ilan_no']
                            if ino not in self.bugun_gorulen:
                                self.bugun_gorulen[ino] = {
                                    'fiyat': item['fiyat'],
                                    'url': item['url'],
                                }
                        self.stats['taranan_sayfa'] += 1
                        self.stats['taranan_ilan'] += len(pg_items)

                    print(f" ✓ {len(self.bugun_gorulen):,} toplam")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f" ⚠️ Hata: {e}")
                continue
    # ──────────────────────────────────────────────────────────────
    #  FİNAL RAPOR
    # ──────────────────────────────────────────────────────────────

    def _print_final_report(self):
        """Detaylı bitiş raporu"""
        s = self.stats

        # Süre hesapla
        elapsed_str = "?"
        if s.get('baslangic'):
            try:
                start = datetime.strptime(s['baslangic'], '%Y-%m-%d %H:%M:%S')
                elapsed = (datetime.now() - start).total_seconds()
                hours = int(elapsed // 3600)
                mins = int((elapsed % 3600) // 60)
                elapsed_str = f"{hours}sa {mins}dk"
            except:
                pass

        # DB'deki güncel toplam
        db_total = "?"
        db_aktif_total = "?"
        db_satilan_total = "?"
        if self.db_conn:
            try:
                self.db_cursor.execute(
                    "SELECT COUNT(*) FROM ilanlar WHERE kaynak = 'arabam'"
                )
                db_total = f"{self.db_cursor.fetchone()[0]:,}"

                self.db_cursor.execute(
                    "SELECT COUNT(*) FROM ilanlar "
                    "WHERE kaynak = 'arabam' AND ilan_durumu = 'aktif'"
                )
                db_aktif_total = f"{self.db_cursor.fetchone()[0]:,}"

                self.db_cursor.execute(
                    "SELECT COUNT(*) FROM ilanlar "
                    "WHERE kaynak = 'arabam' AND ilan_durumu = 'satildi'"
                )
                db_satilan_total = f"{self.db_cursor.fetchone()[0]:,}"
            except:
                pass

        print(f"\n")
        print(f"╔══════════════════════════════════════════════════════════╗")
        print(f"║              📊 GÜNLÜK STOK TAKİP RAPORU               ║")
        print(f"╠══════════════════════════════════════════════════════════╣")
        print(f"║  Taranan sayfa              : {s['taranan_sayfa']:>10,}     ║")
        print(f"║  Sitede bulunan ilan        : {s['taranan_ilan']:>10,}     ║")
        print(f"║                                                        ║")
        print(f"║  🆕 Yeni ilan (DB'ye eklenen): {s['yeni_ilan']:>9,}     ║")
        print(f"║  💰 Fiyatı değişen           : {s['fiyat_degisen']:>9,}     ║")
        print(f"║  👁️  Son görülme güncellenen  : {s['son_gorulme_guncellenen']:>9,}     ║")
        print(f"║  🏷️  Satıldı işaretlenen      : {s['satilan']:>9,}     ║")
        print(f"║  ❌ Hatalı                    : {s['hatali']:>9,}     ║")
        print(f"║                                                        ║")
        print(f"║  🌐 Toplam HTTP istek        : {s['toplam_istek']:>9,}     ║")
        print(f"║  ⏱️  Toplam süre              : {elapsed_str:>12}     ║")
        print(f"║                                                        ║")
        print(f"║  ─── VERİTABANI DURUMU ───                             ║")
        print(f"║  Toplam ilan               : {db_total:>12}     ║")
        print(f"║  Aktif ilan                : {db_aktif_total:>12}     ║")
        print(f"║  Satılan ilan              : {db_satilan_total:>12}     ║")
        print(f"╚══════════════════════════════════════════════════════════╝")
        print(f"🏁 Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



if __name__ == "__main__":
        # ═══ VERİTABANI AYARLARI ═══
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'araba_ilan',
        'user': 'postgres',
        'password': 'sifre123'  # ← KENDİ ŞİFRENİ YAZ
    }

    tracker = ArabamStokTakip(db_config=DB_CONFIG)

    print("\n╔══════════════════════════════════════════════════╗")
    print("║  🚗 ARABAM.COM STOK TAKİP - MOD SEÇİMİ         ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  1. Günlük tam stok takip (5-6 Saat)           ║")
    print("║     → Tüm siteyi tarar, yeni/satılan bulur     ║")
    print("║                                                  ║")
    print("║  2. Sadece yeni ilanlar (Hızlı)                ║")
    print("║     → Son 24 saati çeker, stok takibi yapmaz   ║")
    print("║                                                  ║")
    print("║  3. 🛠️ Eksik Satılanları Kurtar (Saniyeler)    ║")
    print("║     → Sadece veritabanındaki tarihlere bakar   ║")
    print("║     → Hata sonrası yarım kalanı tamamlar       ║")
    print("╚══════════════════════════════════════════════════╝")

    secim = input("\nSeçiminiz (1/2/3): ").strip()

    try:
        if secim == '1':
            tracker.run()
        elif secim == '2':
            tracker.run_sadece_yeniler()
        elif secim == '3':
            tracker.run_satilan_kurtarma()
        else:
            print("❌ Geçersiz seçim! 1, 2 veya 3 girin.")
    except KeyboardInterrupt:
        print("\n⛔ Çıkılıyor...")
    except Exception as e:
        print(f"\n❌ Kritik hata: {e}")
        import traceback
        traceback.print_exc() 