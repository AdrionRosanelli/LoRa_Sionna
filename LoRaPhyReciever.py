class LoRaPhyReceiver:
    """
    Implementação da camada física do receptor LoRa usando Sionna
    """

    def __init__(self,
                 spreading_factor=7,      # SF7 = 128 chips por símbolo
                 bandwidth=125e3,         # 125 kHz
                 coding_rate=1,           # Taxa de codificação
                 num_preamble=8,          # Símbolos de preâmbulo
                 sync_word=0x34,          # Palavra de sincronização
                 crc_on=True,            # Habilitar CRC
                 carrier_freq=None,       # Frequência da portadora (Hz)
                 detection_threshold=0.7): # Limiar de detecção

        self.sf = spreading_factor
        self.bw = bandwidth
        self.cr = coding_rate
        self.num_preamble = num_preamble
        self.sync_word = sync_word
        self.crc_on = crc_on
        self.carrier_freq = carrier_freq
        self.detection_threshold = detection_threshold

        # Parâmetros derivados
        self.n_chips = 2**self.sf  # Número de chips por símbolo
        self.symbol_duration = self.n_chips / self.bw
        self.chip_duration = 1 / self.bw

        # Polinômio CRC-16-CCITT (usado no LoRa)
        self.crc_poly = 0x1021

        # Gerar templates de referência
        self._generate_reference_templates()

    def _generate_reference_templates(self):
        """
        Gera templates de referência para detecção e demodulação
        """
        # Template do preâmbulo (chirp up)
        self.preamble_template, _ = self.generate_chirp_sequence(-self.bw/2)
        
        # Template para down chirp
        self.down_chirp_template, _ = self.generate_chirp_sequence(self.bw/2, down_chirp=1)
        
        # Templates para todos os símbolos possíveis
        self.symbol_templates = {}
        for symbol_val in range(self.n_chips):
            freq_offset = (symbol_val * self.bw) / self.n_chips
            template, _ = self.generate_chirp_sequence(freq_offset)
            self.symbol_templates[symbol_val] = template

    def generate_chirp_sequence(self, initial_frequency=0, down_chirp=0):
        """
        Gera sequência chirp característica do LoRa
        """
        t = np.linspace(0, self.symbol_duration, self.n_chips, endpoint=False)

        # Chirp linear com frequência variando de 0 a BW
        if down_chirp == 0:
            freq_slope = self.bw / self.symbol_duration
        else:
            freq_slope = -self.bw / self.symbol_duration

        chirp = np.exp(1j * 2 * np.pi * (
            initial_frequency * t + 0.5 * freq_slope * t**2))

        return chirp, t

    def apply_carrier_demodulation(self, received_signal, time_vector):
        """
        Remove a modulação da portadora do sinal recebido
        """
        if self.carrier_freq is None:
            return received_signal
        
        # Gera portadora complexa conjugada
        carrier_conj = np.exp(-1j * 2 * np.pi * self.carrier_freq * time_vector)
        
        # Demodula o sinal
        demodulated_signal = received_signal * carrier_conj
        
        return demodulated_signal

    def detect_preamble(self, received_signal):
        """
        Detecta o preâmbulo no sinal recebido usando correlação cruzada
        """
        # Correlação cruzada com o template do preâmbulo
        correlation = correlate(received_signal, self.preamble_template, mode='valid')
        correlation_mag = np.abs(correlation)
        
        # Normalizar correlação
        correlation_normalized = correlation_mag / np.max(correlation_mag)
        
        # Encontrar picos que excedem o limiar
        peak_indices = []
        for i in range(len(correlation_normalized)):
            if correlation_normalized[i] > self.detection_threshold:
                # Verificar se é máximo local
                is_peak = True
                window = 10  # Janela para verificar máximo local
                start = max(0, i - window)
                end = min(len(correlation_normalized), i + window + 1)
                
                for j in range(start, end):
                    if j != i and correlation_normalized[j] > correlation_normalized[i]:
                        is_peak = False
                        break
                
                if is_peak:
                    peak_indices.append(i)
        
        return peak_indices, correlation_normalized

    def synchronize_packet(self, received_signal, preamble_start):
        """
        Sincroniza com o pacote detectando down chirps após o preâmbulo
        """
        # Posição estimada do fim do preâmbulo
        preamble_end = preamble_start + self.num_preamble * self.n_chips
        
        # Região onde esperamos encontrar down chirps
        sync_start = preamble_end + self.n_chips  # Após palavra de sincronização
        sync_region = received_signal[sync_start:sync_start + 3 * self.n_chips]
        
        # Correlacionar com down chirp template
        down_correlation = correlate(sync_region, self.down_chirp_template, mode='valid')
        down_correlation_mag = np.abs(down_correlation)
        
        # Encontrar início dos down chirps
        if len(down_correlation_mag) > 0:
            down_start_idx = np.argmax(down_correlation_mag)
            sync_point = sync_start + down_start_idx
        else:
            # Fallback: usar estimativa baseada no preâmbulo
            sync_point = preamble_end + self.n_chips
        
        return sync_point

    def demodulate_symbol(self, symbol_chirp):
        """
        Demodula um símbolo chirp usando correlação com templates
        """
        max_correlation = 0
        detected_symbol = 0
        
        # Correlacionar com todos os templates possíveis
        for symbol_val, template in self.symbol_templates.items():
            if len(symbol_chirp) != len(template):
                # Ajustar comprimento se necessário
                min_len = min(len(symbol_chirp), len(template))
                correlation = np.abs(np.sum(symbol_chirp[:min_len] * np.conj(template[:min_len])))
            else:
                correlation = np.abs(np.sum(symbol_chirp * np.conj(template)))
            
            if correlation > max_correlation:
                max_correlation = correlation
                detected_symbol = symbol_val
        
        return detected_symbol, max_correlation

    def demodulate_symbol_fft(self, symbol_chirp):
        """
        Demodula símbolo usando método FFT (alternativo)
        """
        # Multiplicar pelo chirp de referência conjugado (down chirp)
        dechirped = symbol_chirp * np.conj(self.down_chirp_template[:len(symbol_chirp)])
        
        # FFT para encontrar o pico de frequência
        fft_result = np.fft.fft(dechirped, self.n_chips)
        fft_mag = np.abs(fft_result)
        
        # Encontrar o índice do pico máximo
        peak_idx = np.argmax(fft_mag)
        
        return peak_idx, fft_mag[peak_idx]

    def extract_symbols_from_packet(self, received_signal, sync_point):
        """
        Extrai símbolos do pacote sincronizado
        """
        symbols = []
        current_pos = sync_point
        
        # Pular down chirps (2.25 símbolos)
        current_pos += int(2.25 * self.n_chips)
        
        # Extrair símbolos enquanto houver dados suficientes
        while current_pos + self.n_chips <= len(received_signal):
            symbol_data = received_signal[current_pos:current_pos + self.n_chips]
            
            # Demodular símbolo
            symbol_value, confidence = self.demodulate_symbol_fft(symbol_data)
            symbols.append((symbol_value, confidence))
            
            current_pos += self.n_chips
        
        return symbols

    def decode_symbols_to_bits(self, symbols):
        """
        Decodifica símbolos para bits
        """
        all_bits = []
        
        for symbol_value, confidence in symbols:
            # Converter símbolo para bits (LSB primeiro)
            symbol_bits = []
            for i in range(self.sf):
                bit = (symbol_value >> i) & 1
                symbol_bits.append(bit)
            
            all_bits.extend(symbol_bits)
        
        return all_bits

    def parse_packet_structure(self, decoded_bits):
        """
        Analisa a estrutura do pacote decodificado
        """
        if len(decoded_bits) < 24:  # Mínimo para header
            return None, None, None, False
        
        # Extrair header (primeiros 3 bytes após sincronização)
        header_bits = decoded_bits[:24]
        header_bytes = []
        
        for i in range(0, 24, 8):
            byte_bits = header_bits[i:i+8]
            byte_val = sum(bit * (2**idx) for idx, bit in enumerate(byte_bits))
            header_bytes.append(byte_val)
        
        payload_length = header_bytes[0]
        coding_rate = header_bytes[1]
        crc_enabled = bool(header_bytes[2])
        
        # Extrair payload
        payload_start = 24
        payload_end = payload_start + (payload_length * 8)
        
        if payload_end > len(decoded_bits):
            return None, None, None, False
        
        payload_bits = decoded_bits[payload_start:payload_end]
        
        # Extrair CRC se habilitado
        crc_received = None
        if crc_enabled and payload_end + 16 <= len(decoded_bits):
            crc_bits = decoded_bits[payload_end:payload_end + 16]
            crc_received = sum(bit * (2**idx) for idx, bit in enumerate(crc_bits))
        
        return payload_bits, crc_received, crc_enabled, True

    def crc16_ccitt(self, data):
        """
        Calcula CRC-16-CCITT para os dados
        """
        if isinstance(data[0], int) and all(isinstance(x, int) and 0 <= x <= 255 for x in data):
            data_bytes = data
        else:
            data_bytes = []
            for i in range(0, len(data), 8):
                byte_bits = data[i:i+8]
                if len(byte_bits) < 8:
                    byte_bits.extend([0] * (8 - len(byte_bits)))
                byte_val = sum(bit * (2**idx) for idx, bit in enumerate(byte_bits))
                data_bytes.append(byte_val)

        crc = 0xFFFF
        for byte in data_bytes:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ self.crc_poly
                else:
                    crc <<= 1
                crc &= 0xFFFF

        return crc ^ 0xFFFF

    def verify_crc(self, payload_bits, received_crc):
        """
        Verifica CRC do payload
        """
        calculated_crc = self.crc16_ccitt(payload_bits)
        return calculated_crc == received_crc

    def receive_lora_packet(self, received_signal, time_vector=None):
        """
        Função principal para receber e decodificar pacote LoRa
        """
        # 1. Demodular portadora se necessário
        if time_vector is not None:
            baseband_signal = self.apply_carrier_demodulation(received_signal, time_vector)
        else:
            baseband_signal = received_signal

        # 2. Detectar preâmbulo
        preamble_indices, correlation = self.detect_preamble(baseband_signal)
        
        if not preamble_indices:
            return None, "Preâmbulo não detectado"

        # 3. Usar o primeiro preâmbulo detectado
        preamble_start = preamble_indices[0]
        
        # 4. Sincronizar com o pacote
        sync_point = self.synchronize_packet(baseband_signal, preamble_start)
        
        # 5. Extrair símbolos
        symbols = self.extract_symbols_from_packet(baseband_signal, sync_point)
        
        if not symbols:
            return None, "Nenhum símbolo detectado"

        # 6. Decodificar símbolos para bits
        decoded_bits = self.decode_symbols_to_bits(symbols)
        
        # 7. Analisar estrutura do pacote
        payload_bits, crc_received, crc_enabled, success = self.parse_packet_structure(decoded_bits)
        
        if not success:
            return None, "Erro na decodificação da estrutura do pacote"

        # 8. Verificar CRC se habilitado
        crc_valid = True
        if crc_enabled and crc_received is not None:
            crc_valid = self.verify_crc(payload_bits, crc_received)

        # Preparar resultado
        result = {
            'payload_bits': payload_bits,
            'payload_bytes': self._bits_to_bytes(payload_bits),
            'crc_valid': crc_valid,
            'symbols_detected': len(symbols),
            'preamble_correlation': correlation,
            'sync_point': sync_point
        }

        return result, "Sucesso" if crc_valid else "CRC inválido"

    def _bits_to_bytes(self, bits):
        """
        Converte lista de bits para bytes
        """
        bytes_list = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            if len(byte_bits) < 8:
                byte_bits.extend([0] * (8 - len(byte_bits)))
            byte_val = sum(bit * (2**idx) for idx, bit in enumerate(byte_bits))
            bytes_list.append(byte_val)
        return bytes_list

    def plot_reception_analysis(self, received_signal, result=None):
        """
        Plota análise da recepção
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Sinal recebido
        axes[0].plot(np.real(received_signal))
        axes[0].set_title('Sinal Recebido (Parte Real)')
        axes[0].set_xlabel('Amostras')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Correlação do preâmbulo
        preamble_indices, correlation = self.detect_preamble(received_signal)
        axes[1].plot(correlation)
        axes[1].axhline(y=self.detection_threshold, color='r', linestyle='--', label='Limiar')
        if preamble_indices:
            for idx in preamble_indices:
                axes[1].axvline(x=idx, color='g', linestyle=':', alpha=0.7)
        axes[1].set_title('Correlação do Preâmbulo')
        axes[1].set_xlabel('Posição')
        axes[1].set_ylabel('Correlação Normalizada')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Espectrograma
        f, t, Sxx = signal.spectrogram(received_signal, fs=self.bw, nperseg=256)
        axes[2].pcolormesh(t, f/1000, 10*np.log10(np.abs(Sxx)), shading='gouraud')
        axes[2].set_title('Espectrograma do Sinal Recebido')
        axes[2].set_xlabel('Tempo (s)')
        axes[2].set_ylabel('Frequência (kHz)')
        
        plt.tight_layout()
        plt.show()

print("Classe LoRaPhyReceiver implementada com sucesso.")