    def load_from_saved_extraction(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Public method to load from saved extraction"""
        return self._load_saved_extraction(file_path)
    
    def get_all_saved_extractions(self) -> List[Dict[str, Any]]:
        """Get list of all saved extractions with metadata"""
        extractions = []
        
        if not os.path.exists(self.content_dir):
            return extractions
        
        for filename in os.listdir(self.content_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.content_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    extractions.append({
                        'extraction_file': filepath,
                        'source_file': data.get('source_file', ''),
                        'file_hash': data.get('file_hash', ''),
                        'extraction_date': data.get('extraction_date', ''),
                        'content_stats': {
                            'text_items': len(data.get('content', {}).get('text', [])),
                            'table_items': len(data.get('content', {}).get('tables', [])),
                            'visual_items': len(data.get('content', {}).get('visuals', []))
                        }
                    })
                except Exception as e:
                    print(f"Error reading extraction file {filename}: {e}")
        
        return extractions
    
    def clear_saved_extractions(self, keep_recent_days: int = 30):
        """Clear old extraction files"""
        if not os.path.exists(self.content_dir):
            return
        
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=keep_recent_days)
        
        removed_count = 0
        for filename in os.listdir(self.content_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.content_dir, filename)
                try:
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_mtime < cutoff_date:
                        os.remove(filepath)
                        removed_count += 1
                except Exception as e:
                    print(f"Error removing old extraction file {filename}: {e}")
        
        print(f"Removed {removed_count} old extraction files")
    
    def process_multiple_documents(self, file_paths: List[str], force_reprocess: bool = False) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Process multiple documents and return combined results"""
        results = {}
        
        for file_path in file_paths:
            try:
                print(f"\nProcessing: {os.path.basename(file_path)}")
                if Config.ENABLE_MULTIPROCESSING:
                    result = self.process_document_parallel(file_path, force_reprocess)
                else:
                    result = self.process_document(file_path, force_reprocess)
                results[file_path] = result
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results[file_path] = {"text": [], "tables": [], "visuals": []}
        
        return results